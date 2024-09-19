#ifndef QRIUS_INCLUDE_GUARD_MULTICAST_RINGBUFF_HPP
#define QRIUS_INCLUDE_GUARD_MULTICAST_RINGBUFF_HPP

#include <perf_utils.hpp>
#include <seqlock.hpp>
#include <uninitialized_storage.hpp>

#include <cstddef>
#include <tuple>
#include <new>
#include <bit>
#include <array>
#include <atomic>
#include <cassert>
#include <type_traits>
#include <functional>
#include <limits>
#include <iostream>

#include <string.h>
#include <sys/mman.h>

namespace qrius
{

template<typename T, std::size_t reader_count_, std::size_t capacity_, bool readers_may_join_late>
class RingBuff;

///
/// This is the first of the two implementations of MultiCast Ringbuffers in this file.
/// Each built for different usecases and are benchmarked to perform better than most implementations
/// that were tested.
///
/// A single writer multiple reader ringbuffer.
/// This is a multicast ringbuffer i.e. every message is effectively a broadcast to every reader.
/// Not the same usecase as load balancing where no reader reads the same message.
///
/// This can also be used as a single writer single reader ringbuffer.
/// It is faster than many single producer single consumer ringbuffers.
///
/// It's very 'fast' with no locks, rmw operations and reduces the cacheline contention
/// between the writer and readers. It's based on seqlocks.
/// Each entry in the ringbuffer is protected by a seqlock which leads to certain interesting properties
/// that makes it a unique fit for certain usecases.
///
/// Writer never blocks here even if we want it to.
/// If the writer laps the reader, reader will lose all data up to the current point.
/// Reader will always start picking up from where the writer is currently - when readers get lapped.
/// This leads to data loss for very slow readers that can't keep up with writer.
/// This is by design and works well for the usecase where a slow reader should not slow
/// down/block the writer and/or other faster readers.
///
/// If the usecase is for the writer to block to prevent lapping the slowest reader, try the
/// next implementation which blocks writer until readers are caught up.
///
/// This may be ok for usecases with a few fast readers along with one or more slower
/// readers that are allowed to lose messages.
///
/// Here readers might lose the older data (that might be stale anyways) while a
/// regular ringbuffer (like the alternate one below) slow reader may have to lose the
/// recent data.
/// For e.g. Handling high throughput real time market data from financial exchanges.
/// Losing older data might be preferable compared to the recent data.
///
/// Scales well to more readers with very less impact to the writer performance.
/// reader_count is the maximum readers to be supported but readers can join
/// in anytime.
/// A slow reader doesn't impact the writer hence doesn't impact the other readers.
///
/// T must satisfy the constraints enforced by Seqlock<T>
/// User code won't compile otherwise.
///
template<typename T, std::size_t reader_count_, std::size_t capacity_>
    requires (reader_count_>0UL &&
              capacity_>0UL &&
              !(capacity_ & (capacity_-1)))
class RingBuff<T, reader_count_, capacity_, true>
{
public:
    static auto constexpr reader_count = reader_count_;
    static auto constexpr capacity = capacity_;

    using ValueType = T;
    using Seqno = typename Seqlock<T>::Seqno;

    constexpr RingBuff() noexcept
        : writer(*this)
    {
        readers.fill(Reader(*this));
        assert(test_cacheline_align(writer));
        assert(test_cacheline_align(readers));
        assert(test_cacheline_align(seq_locks));
    }

    constexpr auto& get_writer() noexcept
    {
        return writer;
    }

    constexpr auto& get_reader(std::size_t at) noexcept
    {
        assert(at < reader_count);
        return readers[at];
    }

    class Writer
    {
    public:
        constexpr Writer(RingBuff& ringbuff)
            : data{0, ringbuff}
        {}

        void emplace(auto&&... args) noexcept
        {
            data.ringbuff.seq_locks[RingBuff::index(data.seqno)].emplace(data.seqno, std::forward<decltype(args)>(args)...);
            ++data.seqno;
        }
    private:
        struct
        {
            Seqno     seqno{0};
            RingBuff& ringbuff;
        }data;
        CachelinePadd<decltype(data)> padd{};
    };

    class Reader
    {
    public:
        Reader() = default; // This should be deleted but makes mayhem with Reader inside an std::array

        constexpr Reader(RingBuff& ringbuff) noexcept
            : data{0, &ringbuff}
        {}

        bool data_available() const noexcept
        {
            assert(data.ringbuff);
            return data.ringbuff->seq_locks[RingBuff::index(data.seqno)].read_ready(data.seqno);
        }

        ///
        /// reads assuming data has become available in the slot
        /// reader is at.
        /// Always ensure data_available() is true before calling this.
        /// It will spin momentarily while a writer is working on this slot.
        /// This mutates the reader state.
        /// i.e. updates the cursor to the next slot to read.
        ///
        T read_data() noexcept
        {
            assert(data.ringbuff);
            auto [result, new_seqno] = data.ringbuff->seq_locks[RingBuff::index(data.seqno)].read(); // T must be trivially copyable
            assert(data.seqno <= new_seqno);
            data.seqno = new_seqno + 1;
            return result;
        }

        ///
        /// Same as read_data() but also provides the expected and actual seqno of the item
        /// that is read. So drops can be tracked with no extra performance cost.
        ///
        /// auto [data, expected_seqno, actual_seqno] = read();
        /// assert(actual_seqno >= expected_seqno);
        /// auto drops = actual_seqno - expected_seqno;
        ///
        std::tuple<T, Seqno, Seqno> read() noexcept
        {
            assert(data.ringbuff);
            auto [result, new_seqno] = data.ringbuff->seq_locks[RingBuff::index(data.seqno)].read();
            assert(data.seqno <= new_seqno);
            auto expected_seqno = data.seqno;
            data.seqno = new_seqno + 1;
            return {result, expected_seqno, new_seqno};
        }
    private:
        struct
        {
            Seqno     seqno{0};
            RingBuff* ringbuff{nullptr}; // TODO:Should have been a reference but forced by std::array.
        }data;
        CachelinePadd<decltype(data)> padd{};
    };

private:
    friend class Writer;
    friend class Reader;
    constexpr static auto index(Seqno seqno) noexcept
    {
        return seqno & (capacity-1);
    }

    alignas(cacheline_align<Writer>) Writer                                 writer;
    alignas(cacheline_align<Reader>) std::array<Reader, reader_count>       readers;
    alignas(cacheline_align<Seqlock<T>>) std::array<Seqlock<T>, capacity>   seq_locks;
    CachelinePadd<decltype(seq_locks)>                                      padd{};
};

template<typename RingBuff, typename Func>
void read_all(typename RingBuff::Reader& reader, Func&& func)
    noexcept(std::is_nothrow_invocable_v<Func, typename RingBuff::ValueType>);

///
/// Multicast queue - single producer multiple consumer.
/// This is another fast implementation with different characteristics.
///
/// Writer & Producer are used interchangeably.
/// Reader & Consumer are used interchangeably.
///
/// Each consumer reads every message i.e. the intention is to not load balance
/// across consumers.
/// Writer will block if the slowest consumer isn't making progress 'fast enough'.
/// Perf tested for high throughput.
/// Theoretically this is likely to be as fast as it gets.
/// If the intention is to minimize variance of the latency distribution,
/// a user may still perform latency specific benchmarks before using this.
///
/// This may be used as a single producer single consumer queue by setting readers = 1.
/// This is still much faster than other SPSC Queues that were part of the
/// benchmarks.
/// No locks or rmw operations.
/// Also minimizes the number of access to the shared cacheline between reader and writer.
/// This makes a huge difference to the throughput compared to say folly or boost spsc queues.
///
/// Note this may not scale well (in terms of throughput) if reader count increases especially
/// for small ringbuffer sizes or when there are slower readers, since writer has to snoop
/// the cachelines of all readers when it gets caught up to the prior snoop.
/// This only happens when writer laps the smallest of the previously snooped positions
/// of the readers.
/// Larger the ringbuffer better the writer scales.
///
/// Seqlock based ringbuffer above has much better reader-count-scaling characteristics.
///
/// API supports batching of the reads and writes.
/// Batching may significantly improve the writer throughput.
/// For e.g. On receiving a burst of traffic over network, there may be more than one
/// message ready to write at a time. Here, writer batching would provide significant improvements
/// in throughput and latency rather writing one message at a time.
/// More details in the Reader and Writer classes.
///
/// Even with no batching the performance measured is significantly higher than other
/// candidates that were benchmarked.
///
/// If reader count is greater than 1, this implementation destroys the elements only
/// in the writer thread when lapping. This is so that we don't have to ref count the readers
/// to prevent rmw operations.
/// This may not be ideal if it's important to release resources that are in T in a 'realtime' fashion
/// For e.g. T owns an fd that needs to released after all readers are done with it (a poor example but conveys the point).
///
/// Note T can not be a C array. Use std::array instead.
/// For the same reasons why std::vector doesn't support C array.
/// Fix https://cplusplus.github.io/LWG/issue3436 discussed here.
///
/// T doesn't have to be Trivially Constructible unlike the ringbuffer based on seqlocks.
//
/// The implementation uses value semantics.
/// The idea is to use this from shared memory directly after an mmap with
/// zero copy semantics in future. I inted to work on this implementation as time permits.
/// C++20 onwards provides various capabilities to bring an object into existence
/// via implicit object creation, std::launder, std::start_lifetime_as etc. without disrupting
/// strict aliasing rules. To have well defined behavior in such a scenario, T must have
//  at least one trivial constructor and must be trivially destructible -
/// though these requirements are not required in this implementation.
///
/// From reading more resources/references for comparison online, in theory, this is a multicast
/// ringbuffer equivalent to a special case of disruptor pattern where One producer is
/// BROADCASTING data to multiple consumers (readers) i.e. each reader receives every item from the producer.
/// https://lmax-exchange.github.io/disruptor/disruptor.html#_throughput_performance_testing
/// This can form the basis of the disruptor's ringbuffer.
/// Would provide same performance characteristics but with more capabilities and flexibilities.
/// I intend to build on this to implement the disruptor based APIs (diamond style producer-consumer).
///
template<typename T, std::size_t reader_count_, std::size_t capacity_>
    requires (reader_count_>0UL &&
              capacity_>0UL &&
              !(capacity_ & (capacity_-1)))
class RingBuff<T, reader_count_, capacity_, false>
{
public:
    static auto constexpr capacity      = capacity_;
    static auto constexpr reader_count  = reader_count_;

    using ValueType                     = T;
    using Seqno                         = std::size_t;

    constexpr RingBuff() noexcept
        : writer(*this)
    {
        readers.fill(Reader(*this));
    }

    ///
    /// Sample usage of the batched writer.
    /// This is the typical workload of the writer thread.
    ///
    /// auto &writer = ringbuf.get_writer();
    /// auto slots = writer.acquire<max_slots>(); // Acquires up to max_slots as available from the ringbuffer.
    ///                                           // User may spin, block or do other useful work untill slots becomes non-zero.
    /// for(auto i=0UL; i!= slots; ++i)
    ///     writer.emplace(args...); // Create the objects on all slots thus acquired.
    ///                              // Emplacing on more slots will break the ringbuffer
    ///                              // Emplacing on less slots is fine but commit() will only
    /// writer.commit();             // commit as many slots that were emplaced.
    //                               // acquire -> emplace -> commit has to happen in this order.
    ///                              // For strong exception safety, commit also must be invoked if emplace
    ///                              // throws (i.e. if T's relevant constructor throws).
    ///
    class Writer
    {
    public:
        constexpr Writer(RingBuff& ringbuff)
            : ringbuff(ringbuff)
        {
            assert(test_cacheline_align(committed_seqno));
            assert(test_cacheline_align(cached_reader_seqno));
        }

        ///
        /// This is a simpler write method that can not be used with batching APIs.
        /// i.e. use this API to write one element at a time or batch api (a combination
        /// of acquire(), emplace(), emplace(), ..., commit()) but not together.
        ///
        /// If batched writes is not your usecase, this alternative might be faster by 20% (from the perf tests).
        ///
        /// Provides strong exception safety guarantee at a minimum and nothrow guranteed if corresponding
        /// Constructor of T doesn't throw
        ///
        bool write(auto&&... args) noexcept(std::is_nothrow_constructible_v<T, decltype(args)...>)
                    requires(reader_count==1UL  ||                  // Can't provide strong exception safety if atleast one of these is not true.
                             std::is_trivially_destructible_v<T> ||
                             std::is_nothrow_constructible_v<T, decltype(args)...>)
        {
            auto seqno = committed_seqno.load(std::memory_order_relaxed);
            if(seqno == cached_reader_seqno + capacity) [[unlikely]]
            {
                cached_reader_seqno = ringbuff.snoop_readers(); // Snoop the readers cachelines only when needed.
            }
            if(seqno == cached_reader_seqno + capacity) [[unlikely]]
            {
                return false;
            }
            if constexpr (reader_count > 1 && !std::is_trivially_destructible_v<T>) // This also makes it broken if construction later throws i.e. we might destroy the same object again
            {
                if(seqno > capacity)
                {
                    ringbuff.storage.destroy_at(RingBuff::index(in_progress_seqno)); // lazy destruction in writer thread when lapping
                }
            }
            ringbuff.storage.construct_at(RingBuff::index(seqno), std::forward<decltype(args)>(args)...);
            committed_seqno.store(seqno+1, std::memory_order_release);
            return true;
        }
        ///
        /// Batched writes.
        /// Acquire up to max_slots for writing
        /// Make sure to always acquire the needed slots for emplacing
        /// and commit all slots acquired in one call afterwards
        /// This is for batching the writer from hitting the reader's
        /// shared state/cacheline often.
        /// This may improve performance significantly.
        /// For e.g. Going from 1 to 32, increases throughput 2.5 times (210M to 510M).
        /// No improvements were observed increasing it further.
        ///
        template<std::size_t max_slots=1UL> requires (max_slots != 0UL)
        std::size_t acquire() noexcept
        {
            if(in_progress_seqno == cached_reader_seqno + capacity) [[unlikely]] //: No improvement noted and may worsen performance when a reader isn't keeping up
            {
                cached_reader_seqno = ringbuff.snoop_readers(); // This is where one slower reader may impact the writer performance.
            }
            return std::min(cached_reader_seqno + capacity - in_progress_seqno, max_slots);
        }

        ///
        /// emplace on the acquired slots only.
        /// i.e. acquire() must return non-zero slots for this function to be invoked.
        /// if reader_count >1, this may invoke the destructor on the prior entry if there was one (i.e. we lapped the ringbuffer)
        ///
        /// So in this implementation, for multiple readers, the construction and destruction takes place in the writer.
        /// For a more realtime destruction (i.e. as soon as all readers complete), this is not right implementation for you.
        /// We may need atomic refcounting for such a scenario which defeats our performance goals.
        ///
        /// Provides strong exception safety guarantee.
        /// If reader_count > 1, relevant constructor of T cannot throw to provide strong exception safety.
        ///
        void emplace(auto &&... args) noexcept (std::is_nothrow_constructible_v<T, decltype(args)...>)
                    requires(reader_count==1UL || // Can't provide strong exception safety if at least one of these are not true.
                             std::is_trivially_destructible_v<T> ||
                             std::is_nothrow_constructible_v<T, decltype(args)...>)
        {
            //
            // if reader_count > 1, we can only destroy in the writer thread
            // Alternative is lifetime management via reference count which must be
            // be expensive.
            // But this forces specific requirements for exception safety of emplace.
            // If reader_count > 1 && the relevant constructor throws, we can't provide
            // the basic exception safety guarantee. So in multiple readers case, the relevant
            // constructor of T can't throw.
            //
            if constexpr (reader_count > 1 && !std::is_trivially_destructible_v<T>)
            {
                if(in_progress_seqno > capacity)
                {
                    ringbuff.storage.destroy_at(RingBuff::index(in_progress_seqno)); // lazy destruction in writer thread when lapping
                }
            }
            ringbuff.storage.construct_at(RingBuff::index(in_progress_seqno), std::forward<decltype(args)>(args)...);
            ++in_progress_seqno;
        }

        ///
        /// commit() after writing (emplacing) to all slots acquire()-d.
        ///
        /// For strong exception safety, make sure to invoke commit if
        /// emplace throws (in case T has a throwing constructor) -
        /// to make sure that previous objects written using prior emplace-s
        /// will be recorded into the ringbuffer.
        ///
        void commit() noexcept
        {
            committed_seqno.store(in_progress_seqno, std::memory_order_release);
        }

        void test_alignments() const noexcept
        {
            assert(test_cacheline_align(cached_reader_seqno));
            assert(test_cacheline_align(committed_seqno));
        }

    private:
        friend class RingBuff;

        Seqno seqno() const noexcept
        {
            return committed_seqno.load(std::memory_order_acquire);
        }

        Seqno                                       cached_reader_seqno{0};
        Seqno                                       in_progress_seqno{0};
        RingBuff                                    &ringbuff;
        alignas(cacheline_size) std::atomic<Seqno>  committed_seqno{0};
        CachelinePadd<decltype(committed_seqno)>    padd{};
    };

    class Reader
    {
    public:
        constexpr Reader() noexcept = default;

        constexpr Reader(RingBuff& ringbuff) noexcept
            : ringbuff(&ringbuff)
        {
            assert(test_cacheline_align(committed_seqno));
            assert(test_cacheline_align(cached_writer_seqno));
        }

        constexpr Reader(Reader const&) = default;

        Reader& operator = (Reader const& rhs) noexcept // TODO: Remove/Fix - Forced by std::array.
        {
            cached_writer_seqno = rhs.cached_writer_seqno;
            in_progress_seqno = rhs.in_progress_seqno;
            ringbuff = rhs.ringbuff;
            committed_seqno.store(rhs.committed_seqno.load());
            return *this;
        }

        ///
        /// Increase max_slots only if there is proof that reader isn't keeping up
        /// or to trade-off writer's throughput with reader's latency.
        /// Otherwise reader will end up contending on writer-seqno-cacheline while spinning waiting for more data.
        /// This is a case of reader busy waiting for writer.
        /// For e.g. In the perf test increasing this to 32 while keeping writer max_slots to 1
        /// halves the writer's throughput.
        ///
        template<std::size_t max_slots=1UL> requires (max_slots != 0UL)
        std::size_t acquire() noexcept
        {
            assert(ringbuff);
            if(in_progress_seqno == cached_writer_seqno) //[[likely]] if reader is always ready/up with the writer.
            {
                cached_writer_seqno = ringbuff->snoop_writer();
            }
            return std::min(max_slots, cached_writer_seqno - in_progress_seqno);
        }

        ///
        /// non-mutable read.
        /// Must invoke pop to finally mark the read transaction as complete.
        ///
        T const& read() const noexcept
        {
            assert(ringbuff);
            return ringbuff->storage[RingBuff::index(in_progress_seqno)];
        }

        ///
        /// pop() must be invoked to make sure that reader progresses after read.
        ///
        void pop() noexcept
        {
            if constexpr(reader_count == 1)
            {
                assert(ringbuff);
                ringbuff->storage.destroy_at(RingBuff::index(in_progress_seqno));
            }
            ++in_progress_seqno;
        }

        void commit() noexcept
        {
            committed_seqno.store(in_progress_seqno, std::memory_order_release);
        }

        void test_alignments() const noexcept
        {
            assert(test_cacheline_align(cached_writer_seqno));
            assert(test_cacheline_align(committed_seqno));
        }
    private:
        friend class RingBuff;

        Seqno seqno() const noexcept
        {
            return committed_seqno.load(std::memory_order_acquire);
        }

        Seqno                                       cached_writer_seqno{0};
        Seqno                                       in_progress_seqno{0};
        RingBuff                                   *ringbuff{nullptr}; // TODO: Reference preferred but std::array forces this.
        alignas(cacheline_size) std::atomic<Seqno>  committed_seqno{0};
        CachelinePadd<decltype(committed_seqno)>    padd{};
    };

    constexpr Writer& get_writer() noexcept
    {
        return writer;
    }

    constexpr Reader& get_reader(std::size_t at) noexcept
    {
        assert(at < reader_count);
        return readers[at];
    }

    ~RingBuff() noexcept
    {
        if constexpr (std::is_trivially_destructible_v<T>) return;

        if constexpr(reader_count == 1UL)
        {
            auto empty_func = [](ValueType const&) noexcept {};
            // May in a future revision, these template parameters be auto deduced!!
            read_all<std::decay_t<decltype(*this)>, decltype(empty_func)>(get_reader(0UL), std::move(empty_func));
        }
        else
        {
            auto& writer = get_writer();
            writer.commit();
            auto seqno = index(writer.seqno());
            for(auto i=0UL; i != seqno; ++i)
            {
                storage.destroy_at(i);
            }
        }
    }
private:
    Seqno snoop_writer() const noexcept
    {
        return writer.seqno();
    }

    Seqno snoop_readers() const noexcept
    {
        // For a single reader, this function gets collapsed to return reader.seqno();
        auto min = std::numeric_limits<Seqno>::max();
        for(auto &reader : readers)
        {
            min = std::min(min, reader.seqno());
        }
        return min;
    }

    friend class Writer;
    friend class Reader;
    constexpr static auto index(std::size_t seqno) noexcept
    {
        return seqno & (capacity-1);
    }

    alignas(cacheline_align<Writer>) Writer                             writer;
    alignas(cacheline_align<Reader>) std::array<Reader, reader_count>   readers;
    alignas(cacheline_align<T>) UninitializedArray<T, capacity, true> storage;
    CachelinePadd<decltype(storage)>                                    padd{};
};

template<typename RingBuff, typename Func>
inline void read_all(typename RingBuff::Reader& reader, Func&& func)
    noexcept(std::is_nothrow_invocable_v<Func, typename RingBuff::ValueType>)
{
    auto slots = reader.acquire();
    while(slots)
    {
        for(auto slot=0UL; slot != slots; ++slots)
        {
            std::invoke(func, reader.read());
            reader.pop();
        }
        reader.commit();
        slots = reader.acquire();
    }
}

}
#endif