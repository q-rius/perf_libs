#ifndef QRIUS_INCLUDE_GUARD_MULTICAST_RINGBUFF_HPP
#define QRIUS_INCLUDE_GUARD_MULTICAST_RINGBUFF_HPP

#include "perf_utils.hpp"
#include "seqlock.hpp"

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

///
/// All ring buffer implementations below are value types with compile time sizes.
/// It just works for all my common use cases including shared memory use cases too.
///
template<typename T, std::size_t size, bool force_page_fault_at_init=true> requires (size > 0UL)
class UninitializedStorage
{
public:
    constexpr UninitializedStorage() noexcept
    {
        assert(test_alignment(storage[0], std::max(alignof(T), cacheline_size)));
        if constexpr(force_page_fault_at_init)
        {
            for(auto i=0UL; i<sizeof(storage); i+=page_size)
            {
                storage[i] = std::byte{'\0'};
            }
            storage[sizeof(storage)-1] = std::byte{'\0'};
        }
    }
    constexpr T& operator [] (std::size_t at) noexcept
    {
        return *std::launder(std::bit_cast<T*>(&storage[at*sizeof(T)]));
    }

    constexpr T const& operator [] (std::size_t at) const noexcept
    {
        return *std::launder(std::bit_cast<T const*>(&storage[at*sizeof(T)]));
    }

    constexpr void construct_at(std::size_t at, auto&&... args) noexcept(std::is_nothrow_constructible_v<T, decltype(args)...>)
    {
        std::construct_at(std::launder(std::bit_cast<T*>(&storage[at*sizeof(T)])), std::forward<decltype(args)>(args)...);
    }

    constexpr void destroy_at(std::size_t at) noexcept
    {
        std::destroy_at(std::launder(std::bit_cast<T*>(&storage[at*sizeof(T)])));
    }
private:
    alignas(alignof(T)) std::byte storage[sizeof(T)*size];
};

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
/// It's very 'fast' with no locks, rmw operations and reduces the possibility of cacheline contention
/// the between the writer and readers.
/// It's based on seqlocks.
/// Each entry in the ringbuffer is protected by a seqlock which leads to certain interesting properties
/// that makes it a unique fit for certain usecases.
///
/// Writer never blocks here. Not even if you want it to.
/// If the writer laps the reader, reader will LOSE all data up to the current point and
/// will start picking up where the writer is currently.
/// This leads to data loss for very slow readers that can't keep up with writer.
/// There is no way to prevent this due to the way it designed and the usecase that a slow reader
/// should not slow down or block the writer or other readers.
/// If the usecase is for the writer to block to prevent lapping the slowest reader, try the
/// next implementation which optionally allows writer to wait until readers are caught up.
///
/// This may be ok for usecases with a few fast readers along with a slower reader which might be
/// ok to lose messages if it cannot keep up.
///
/// Here readers might lose the older data (that might be stale anyways) while in the
/// regular ringbuffer (like the alternate one below) readers will be forced to lose the
/// recent data.
/// For e.g. Handling high throughput real time market data from financial exchanges.
/// Losing older data might be preferable compared to the recent data.
///
/// Scales well to more readers with very less impact to the writer performance.
/// reader_count is the maximum readers to be supported but readers can join
/// in anytime.
///
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
        assert(test_alignment(writer, cacheline_size));
        assert(test_alignment(readers, cacheline_size));
        assert(test_alignment(seq_locks, cacheline_size));
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

        constexpr void emplace(auto&&... args) noexcept
        {
            data.ringbuff.seq_locks[data.ringbuff.index(data.seqno)].emplace(data.seqno, std::forward<decltype(args)>(args)...);
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

        constexpr bool data_available() const noexcept
        {
            assert(data.ringbuff);
            return data.ringbuff->seq_locks[data.ringbuff->index(data.seqno)].read_ready(data.seqno);
        }

        ///
        /// reads assuming data has become available in the slot
        /// reader is at.
        /// Always ensure data_available() is true before calling this.
        /// It will spin momentarily while a writer is working on this slot.
        /// This mutates the reader state.
        /// i.e. updates the cursor to the next slot to read.
        ///
        constexpr T read_data() noexcept
        {
            assert(data.ringbuff);
            auto [result, new_seqno] = data.ringbuff->seq_locks[data.ringbuff->index(data.seqno)].read(); // T must be trivially copyable
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
        constexpr std::tuple<T, Seqno, Seqno> read() noexcept
        {
            assert(data.ringbuff);
            auto [result, new_seqno] = data.ringbuff->seq_locks[data.ringbuff->index(data.seqno)].read();
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
    constexpr auto index(Seqno seqno) const noexcept
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
///
/// No latency benchmarks yet. Theoretically this should be as fast as it gets.
/// If the intention is to minimize variance of the latency distribution,
/// you might still want to benchmark before using this.
///
/// You may use it as single producer single consumer queue by setting readers = 1.
/// This is still much faster than other SPSC Queues that were part of the
/// benchmarks.
/// No locks, rmw operations.
/// Also minimizes the number of access to the shared cacheline between reader and writer.
/// This makes a huge difference to the throughput compared to say folly or boost spsc queues.

/// Note this may not scale well in terms of throughput if consumer count increases especially
/// for small ringbuffer sizes, since writer has to snoop the cachelines of all readers.
/// This only happens when writer laps the smallest of the previously snooped positions
/// of the readers.
///
/// Larger the ringbuffer better the writer scales.
///
/// API supports batching of the reads and writes.
/// Batching may significantly improve the writer throughput.
/// For e.g. if you receive a burst of traffic over network, you may have
/// more than one message ready to write at a time, you may use writer batching
/// at that point providing significant improvement in throughput and latency.
/// More details in the Reader and Writer classes.
///
/// Even with no batching the performance measured is significantly higher than other
/// candidates that were benchmarked.
///
/// If reader count is greater than 1, this implementation destroys the elements only
/// in the writer thread, when lapping since we don't ref count the readers to prevent r-m-w operations.
/// This can be a big disadvantage if we need to release resources that are in T in a 'realtime' fashion
/// For e.g. T owns an fd that needs to released after all readers are done with it (a poor example but conveys the point).
///
/// Note T can not be a C array. Use std::array instead.
/// For the same reasons why std::vector doesn't support C array.
/// Fix https://cplusplus.github.io/LWG/issue3436 discussed here.
///
/// The implementation uses value semantics. The idea is to use this from shared memory
/// directly after an mmap with zero copy semantics in future.
/// C++20 onwards provides various capabilities to bring an object into existence
/// via implicit object creation, std::launder, std::start_lifetime_as etc.
/// So to have well defined behavior T must have at least one trivial constructor and
/// must be trivially destructible (though these requirements are not enforced currently).
///
/// From reading more resources/references for comparison online, in theory, this is a multicast
/// ringbuffer equivalent to a special case of disruptor pattern where One producer is
/// BROADCASTING data to multiple consumers (readers) i.e. each reader receives every item from the producer.
/// https://lmax-exchange.github.io/disruptor/disruptor.html#_throughput_performance_testing
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
    ///
    /// auto writer = ringbuf.get_writer();
    /// auto slots = writer.acquire(); // User may spin, block or do other useful work untill slots becomes non-zero
    /// for(auto i=0UL; i!= slots; ++i)
    ///     writer.emplace(); // Create your object on all slots thus acquired.
    ///                       // Emplacing on more slots will break the ringbuffer
    ///                       // Emplacing on less slots is fine but commit() will only
    /// commit as many slots that were emplaced.
    /// writer.commit(); // acquire emplace commit has to happen in this order.
    ///
    class Writer
    {
    public:
        constexpr Writer(RingBuff& ringbuff)
            : ringbuff(ringbuff)
        {
            assert(test_alignment(committed_seqno, cacheline_size));
            assert(test_alignment(cached_reader_seqno, cacheline_size));
        }

        ///
        /// This is a simpler write method that can not be used with batching APIs.
        /// i.e. use this API to write one element at a time or a
        /// combination of acquire(), emplace(), emplace(), ..., commit()
        /// but not together.
        ///
        /// If batched writes is not your usecase, this alternative might be faster by 20% (from the perf tests).
        ///
        /// Provides strong exception safety guarantee at a minimum and nothrow guranteed if corresponding
        /// Constructor of T doesn't throw
        ///
        constexpr bool write(auto&&... args) noexcept(std::is_nothrow_constructible_v<T, decltype(args)...>)
                    requires(reader_count==1UL  || std::is_nothrow_constructible_v<T, decltype(args)...>)
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
                    ringbuff.storage.destroy_at(ringbuff.index(in_progress_seqno)); // lazy destruction in writer thread when lapping
                }
            }
            ringbuff.storage.construct_at(ringbuff.index(seqno), std::forward<decltype(args)>(args)...);
            committed_seqno.store(seqno+1, std::memory_order_release);
            return true;
        }
        ///
        /// Batched writes.
        /// Acquire up to max_slots for writing
        /// Make sure to always acquire the needed slots for emplacing
        /// and commit all slots acquired in one call afterwards
        /// This is for batching the producer from hitting the reader's
        /// shared state/cacheline often.
        ///
        template<std::size_t max_slots=1UL> requires (max_slots != 0UL)// Note: Going from 1 to 32, increases throughput 2.5 times (210M to 510M) and no further improvements afterwards.
        constexpr std::size_t acquire() noexcept
        {
            if(in_progress_seqno == cached_reader_seqno + capacity) [[unlikely]] //: No improvement noted and may worsen performance when a reader isn't keeping up
            {
                cached_reader_seqno = ringbuff.snoop_readers(); // This is where one slower reader may impact the writer badly.
            }
            return std::min(cached_reader_seqno + capacity - in_progress_seqno, max_slots);
        }

        ///
        /// emplace on the acquired slots only.
        /// if reader_count >1, this may invoke the destructor on the prior entry if there was one (i.e. we lapped the ringbuffer)
        ///
        /// So in this implementation, for multiple readers, the construction and destruction takes place in the writer.
        /// For a more realtime destruction (i.e. as soon as all readers complete), this is not right implementation for you.
        /// We may need atomic refcounting for such a scenario which defeats our performance goals.
        ///
        /// Provides strong exception safety guarantee.
        ///
        constexpr void emplace(auto &&... args) noexcept (std::is_nothrow_constructible_v<T, decltype(args)...>)
                    requires(reader_count==1UL  || std::is_nothrow_constructible_v<T, decltype(args)...>)
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
                    ringbuff.storage.destroy_at(ringbuff.index(in_progress_seqno)); // lazy destruction in writer thread when lapping
                }
            }
            ringbuff.storage.construct_at(ringbuff.index(in_progress_seqno), std::forward<decltype(args)>(args)...);
            ++in_progress_seqno;
        }

        ///
        /// commit() after writing (emplacing) to all slots acquire()-d.
        ///
        constexpr void commit() noexcept
        {
            committed_seqno.store(in_progress_seqno, std::memory_order_release);
        }

        void test_alignments() const noexcept
        {
            assert(test_alignment(cached_reader_seqno, cacheline_size));
            assert(test_alignment(committed_seqno, cacheline_size));
        }

    private:
        friend class RingBuff;

        constexpr Seqno seqno() const noexcept
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
            assert(test_alignment(committed_seqno, cacheline_size));
            assert(test_alignment(cached_writer_seqno, cacheline_size));
        }

        constexpr Reader(Reader const&) = default;

        constexpr Reader& operator = (Reader const& rhs) noexcept //Forced by std::array.
        {
            cached_writer_seqno = rhs.cached_writer_seqno;
            in_progress_seqno = rhs.in_progress_seqno;
            ringbuff = rhs.ringbuff;
            committed_seqno.store(rhs.committed_seqno.load());
            return *this;
        }

        ///
        /// Increase max_slots only if you have proof that reader isn't keeping up
        /// or if you care less about writer's throughput but more about reader's latency.
        /// Otherwise reader will end up contending on writer-seqno-cacheline while spinning waiting for more data.
        /// This is a case of reader busy waiting for writer.
        /// For e.g. In the perf test increasing this to 32 while keeping writer max_slots to 1
        /// halves the writer's throughput.
        ///
        template<std::size_t max_slots=1UL> requires (max_slots != 0UL)
        constexpr std::size_t acquire() noexcept
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
        /// Must invoke pop after completing the read.
        ///
        /// Mutable version of this can be provided if there is a usecase like
        /// in disruptor pattern where parts of T is modified independently by
        /// different readers.
        ///
        constexpr T const& read() const noexcept
        {
            assert(ringbuff);
            return ringbuff->storage[ringbuff->index(in_progress_seqno)];
        }

        ///
        /// pop() must be invoked to make sure that reader progresses after read.
        ///
        constexpr void pop() noexcept
        {
            if constexpr(reader_count == 1)
            {
                assert(ringbuff);
                ringbuff->storage.destroy_at(ringbuff->index(in_progress_seqno));
            }
            ++in_progress_seqno;
        }

        constexpr void commit() noexcept
        {
            committed_seqno.store(in_progress_seqno, std::memory_order_release);
        }

        void test_alignments() const noexcept
        {
            assert(test_alignment(cached_writer_seqno, cacheline_size));
            assert(test_alignment(committed_seqno, cacheline_size));
        }
    private:
        friend class RingBuff;

        constexpr Seqno seqno() const noexcept
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
    constexpr Seqno snoop_writer() const noexcept
    {
        return writer.seqno();
    }

    constexpr Seqno snoop_readers() const noexcept
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
    constexpr auto index(std::size_t seqno) const noexcept
    {
        return seqno & (capacity-1);
    }

    alignas(cacheline_align<Writer>) Writer                             writer;
    alignas(cacheline_align<Reader>) std::array<Reader, reader_count>   readers;
    alignas(cacheline_align<T>) UninitializedStorage<T, capacity, true> storage;
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