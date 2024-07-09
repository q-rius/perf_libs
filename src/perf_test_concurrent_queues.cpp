#include "multicast_ringbuff.hpp"
#include "perf_utils.hpp"
#include "perf_test.hpp"
#include "external_for_bench/folly/ProducerConsumerQueue.h"
#include "external_for_bench/rigtorp/SPSCQueue.h"

#include <atomic>
#include <limits>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <random>
#include <concepts>

using namespace qrius;
struct InvalidReadException: std::runtime_error
{
    using std::runtime_error::runtime_error;
};

///
/// Here we attempt a comprehensive set of throughput tests for data structures
/// that are shared across threads - writer vs readers.
/// All tests have been performed on Linux x86_64.
/// 1. Simple local load, store of std::uint64_t - to see the memory parallelism in action.
/// 2. Shared load, store of std::uint64_t
/// 3. Seqlock - single writer multiple readers
/// 4. Our performance optimized Seqlock based Multicast RingBuffer that never blocks the writer.
/// 5. Our performance optimized Multicast RingBuffer where writer can wait for readers to catch up.
/// 6. Compare Known fast implementations of Single Producer Single Consumer Ringbuffers- refer external_for_bench/ directory.
///

template<typename T> requires (std::atomic<T>::is_always_lock_free)
struct AtomicVar
{
    alignas(qrius::cacheline_align<T>) std::atomic<T> var;
    qrius::CachelinePadd<decltype(var)> padd{};
};

template<typename T>
struct AtomicPair
{
    AtomicVar<T> store_var{0};
    AtomicVar<T> load_var{0};
};

template<typename T, auto readers=1UL>
void perf_test_local_store_load(qrius::CpuSet const& cpu_set, std::size_t test_iters) requires(std::is_integral_v<T>)
{
    std::cout << "\nlocal unshared atomic load store with no fences test_iterations=" << test_iters << '\n';
    T rand_val = std::random_device{}();

    auto construct_func = [=]()
    {
        auto atomic_pair_ptr = std::make_unique<AtomicPair<T>>();
        assert(qrius::test_cacheline_align(*atomic_pair_ptr));
        assert(qrius::test_cacheline_align(atomic_pair_ptr->store_var));
        assert(qrius::test_cacheline_align(atomic_pair_ptr->load_var));
        static_assert(sizeof(AtomicPair<T>) == 2*qrius::cacheline_size);
        static_assert(alignof(AtomicPair<T>) == qrius::cacheline_size);
        atomic_pair_ptr->load_var.var = rand_val;
        return atomic_pair_ptr;
    };
    auto emplace_func = [](auto& atomic_pair,
                           std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
            atomic_pair.store_var.var.store(item, std::memory_order_relaxed);
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [=](auto& atomic_pair,
                         std::size_t reader_index[[maybe_unused]],
                         std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
            if(atomic_pair.load_var.var.load(std::memory_order_relaxed) != rand_val) [[unlikely]] throw InvalidReadException{"From load local load test"};
        }
        return std::pair{test_iters, 0UL};
    };
    analyze_result<T, 1UL, readers+1>(perf_test<decltype(construct_func),
                                                decltype(emplace_func),
                                                decltype(read_func),
                                                readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<typename T, auto readers=1UL>
void perf_test_shared_store_load(qrius::CpuSet const& cpu_set, std::size_t test_iters) requires(std::is_integral_v<T>)
{
    std::cout << "\nshared atomic load store with no fences test_iterations=" << test_iters << '\n';

    auto construct_func = []()
    {
        auto atomic_var_ptr = std::make_unique<AtomicVar<T>>();
        assert(qrius::test_cacheline_align(*atomic_var_ptr));
        return atomic_var_ptr;
    };
    auto emplace_func = [](auto& atomic_var, std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
            atomic_var.var.store(item, std::memory_order_release);
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [](auto& atomic_var, std::size_t reader_index[[maybe_unused]], std::size_t test_iters) __attribute__((noinline, hot))
    {
        assert(test_iters > 0UL);
        auto completed_reads = 0UL;
        while(atomic_var.var.load(std::memory_order_acquire) != test_iters - 1)
        {
            ++completed_reads;
        }
        return std::pair{completed_reads, completed_reads > test_iters ? completed_reads - test_iters : 0UL};
    };
    analyze_result<T,
                   1UL,
                   readers+1>(perf_test<decltype(construct_func),
                                        decltype(emplace_func),
                                        decltype(read_func),
                                        readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

///
/// Throughput test under heavy cacheline contention.
/// i.e. it forces out the maximum possible cache coherence
/// traffic across two cachelines under no locks or rmw instruction.
/// i.e. just ping pong of a load-store under the acquire-release semantic.
///
/// This proves the case for avoiding access to the shared cacheline as much as possible.
/// Hence the case for batching writes/reads when looking for high throughput.
///
template<typename T>
void perf_test_ping_pong_store_load(qrius::CpuSet const& cpu_set, std::size_t test_iters) requires(std::is_integral_v<T>)
{
    std::cout << "\nping pong store load with no fences test_iterations=" << test_iters << '\n';
    auto construct_func = [=]()
    {
        auto atomic_pair_ptr = std::make_unique<AtomicPair<T>>();
        atomic_pair_ptr->store_var.var.store(1);
        atomic_pair_ptr->load_var.var.store(1);
        assert(qrius::test_cacheline_align(*atomic_pair_ptr));
        assert(qrius::test_cacheline_align(atomic_pair_ptr->store_var));
        assert(qrius::test_cacheline_align(atomic_pair_ptr->load_var));
        static_assert(sizeof(AtomicPair<T>) == 2*qrius::cacheline_size);
        static_assert(alignof(AtomicPair<T>) == qrius::cacheline_size);
        return atomic_pair_ptr;
    };
    auto emplace_func = [](auto& atomic_pair,
                           std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
            atomic_pair.store_var.var.store(item, std::memory_order_release);
            while(atomic_pair.load_var.var.load(std::memory_order_acquire) != item) asm("pause");
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [=](auto& atomic_pair,
                         std::size_t reader_index[[maybe_unused]],
                         std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
            while(atomic_pair.store_var.var.load(std::memory_order_acquire) != item) asm("pause");
            atomic_pair.load_var.var.store(item, std::memory_order_release);
        }
        return std::pair{test_iters, 0UL};
    };
    analyze_result<T>(perf_test<decltype(construct_func),
                                decltype(emplace_func),
                                decltype(read_func),
                                1UL>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<typename T>
struct PaddedSeqlock
{
    using Seqlock = qrius::Seqlock<T>;
    alignas(qrius::cacheline_align<Seqlock>) Seqlock seqlock;
    qrius::CachelinePadd<Seqlock> padd;
};

template<typename T, std::size_t readers=1UL>
void perf_test_qrius_seqlock(qrius::CpuSet const& cpu_set, std::size_t test_iters) requires (std::is_constructible_v<T, std::size_t> &&
                                                                                             std::totally_ordered<T>)
{
    std::cout << "\nseqlock test_iterations=" << test_iters << '\n';

    using Seqlock = PaddedSeqlock<T>;
    auto construct_func = []()
    {
        auto seqlock_ptr = std::make_unique<Seqlock>();
        static_assert(alignof(Seqlock) >= qrius::cacheline_size);
        static_assert(!(sizeof(Seqlock) % qrius::cacheline_size));
        assert(qrius::test_cacheline_align(*seqlock_ptr));
        return seqlock_ptr;
    };
    auto emplace_func = [](auto& padded_seqlock,
                           std::size_t test_iters) __attribute__((noinline, hot)) // We don't want this inlined lest compiler might have more ideas to reorder these across timing calls.
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
            padded_seqlock.seqlock.emplace(item, item);
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [](auto& padded_seqlock,
                        std::size_t reader_index[[maybe_unused]],
                        std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto item = 0UL;
        auto wasted_ops = 0UL;
        auto completed_ops = 0UL;
        while(item < test_iters)
        {
            while(!padded_seqlock.seqlock.read_ready(item))
            {
                ++wasted_ops;
                asm("pause");
            }
            auto [result, seqno] = padded_seqlock.seqlock.read();
            ++completed_ops; // We need actual completed reads (since writer just drops when readers don't keep up.)
            if(result < item) [[unlikely]] throw InvalidReadException("Seqlock read unexpected value. Thread race/bug.");
            item = seqno + 1;
        }
        return std::pair{completed_ops, wasted_ops};
    };
    analyze_result<T,
                   1UL,
                   readers+1>(perf_test<decltype(construct_func),
                                        decltype(emplace_func),
                                        decltype(read_func),
                                        readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<typename T, bool huge_pages>
auto make_helper(auto&&... args)
{
    if constexpr(huge_pages)
    {
        auto t_ptr = qrius::make_unique_on_huge_pages<T>(std::forward<decltype(args)>(args)...);
        assert(qrius::test_cacheline_align(*t_ptr));
        return t_ptr;
    }
    else
    {
        auto t_ptr = std::make_unique<T>(std::forward<decltype(args)>(args)...);
        assert(qrius::test_cacheline_align(*t_ptr));
        return t_ptr;
    }
}

template<typename T, std::size_t capacity, std::size_t readers=1UL, bool huge_pages=false>
void perf_test_seqlock_ringbuff(qrius::CpuSet const& cpu_set, std::size_t test_iters) requires(std::is_constructible_v<T, std::size_t>)
{
    std::cout << "\nseqlock ringbuff test_iterations=" << test_iters << '\n';
    using MCRingBuff = qrius::RingBuff<T, !readers ? 1UL : readers, capacity, true>;
    auto construct_func = []()
    {
        return make_helper<MCRingBuff, huge_pages>();
    };
    auto emplace_func = [](auto& ring_buff,
                           std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto& writer = ring_buff.get_writer();
        for(auto item=0UL; item != test_iters; ++item)
        {
            writer.emplace(item);
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [](auto& ring_buff,
                        std::size_t reader_index,
                        std::size_t test_iters) __attribute__((noinline, hot))
    {
        if constexpr (!readers) return std::pair{0UL, 0UL};
        auto &reader = ring_buff.get_reader(reader_index);
        auto item = 0UL;
        auto wasted_ops = 0UL;
        auto completed_ops = 0UL;
        while(item < test_iters)
        {
            while(!reader.data_available())
            {
                ++wasted_ops;
                asm("pause");
            }
            auto result = reader.read_data();
            if(result < T{item}) [[unlikely]] throw InvalidReadException("SeqlockRingBuff bug/thread race");
            item = result + 1;
            ++completed_ops;
        }
        return std::pair{completed_ops, wasted_ops};
    };
    analyze_result<typename MCRingBuff::ValueType,
                   capacity,
                   readers+1>(perf_test<decltype(construct_func),
                                        decltype(emplace_func),
                                        decltype(read_func),
                                        readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<std::size_t capacity>
struct Region
{
    alignas(qrius::cacheline_size) std::array<std::size_t, capacity> elements{};
    qrius::CachelinePadd<decltype(elements)>                         padd{};
};

template<std::size_t capacity, std::size_t readers=1UL>
void perf_test_shared_region(qrius::CpuSet const& cpu_set, std::size_t test_iters) requires(!(capacity&(capacity-1)))
{
    test_iters = std::min(capacity, test_iters);
    std::cout << "\nlock free shared region test_iterations=" << test_iters << '\n';
    auto construct_func = [=]() __attribute__((noinline))
    {
        auto shared_region_ptr = std::make_unique<Region<capacity>>();
        shared_region_ptr->elements.fill(std::numeric_limits<std::size_t>::max());
        assert(qrius::test_cacheline_align(*shared_region_ptr));
        static_assert(!(sizeof(Region<capacity>)%qrius::cacheline_size));
        using namespace std::chrono_literals;
        qrius::clflush(*shared_region_ptr);
        std::this_thread::sleep_for(1s);
        return shared_region_ptr;
    };
    auto emplace_func = [](auto& shared_region,
                           std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
           shared_region.elements[item] = item;
           //asm("SFENCE"); // This seems to exactly reproduce the performance regression in multicast ringbuffer
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [=](auto& shared_region,
                         std::size_t reader_index[[maybe_unused]],
                         std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto wasted_reads = 0UL;
        for(auto item=0UL; item != test_iters; ++item)
        {
            while(shared_region.elements[item] == std::numeric_limits<std::size_t>::max())
            {
                asm("pause");
                ++wasted_reads;
            }
            if(shared_region.elements[item] != item) [[unlikely]] throw InvalidReadException("Read unexpected value. Thread race/bug.");
        }
        return std::pair{test_iters, wasted_reads};
    };
    analyze_result<std::size_t>(perf_test<decltype(construct_func),
                                decltype(emplace_func),
                                decltype(read_func),
                                readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<std::size_t capacity>
struct RegionSeqno
{
    alignas(qrius::cacheline_size) std::atomic<std::size_t> seqno{0UL};
    alignas(qrius::cacheline_size) std::array<std::size_t, capacity> elements;
    qrius::CachelinePadd<decltype(elements)> padd{};
};

template<std::size_t capacity, std::size_t readers=1UL>
void perf_test_shared_region_seqno(qrius::CpuSet const& cpu_set,
                                   std::size_t test_iters) requires(!(capacity&(capacity-1)))
{
    test_iters = std::min(capacity, test_iters);
    std::cout << "\nlock free shared region with seqno test_iterations=" << test_iters << '\n';
    auto construct_func = [=]() __attribute__((noinline))
    {
        auto shared_region_ptr = std::make_unique<RegionSeqno<capacity>>();
        shared_region_ptr->elements.fill(std::numeric_limits<std::size_t>::max());
        assert(qrius::test_cacheline_align(*shared_region_ptr));
        static_assert(!(sizeof(RegionSeqno<capacity>)%qrius::cacheline_size));
        using namespace std::chrono_literals;
        qrius::clflush(*shared_region_ptr);
        std::this_thread::sleep_for(1s);
        return shared_region_ptr;
    };
    auto emplace_func = [](auto& shared_region, std::size_t test_iters) __attribute__((noinline, hot))
    {
        for(auto item=0UL; item != test_iters; ++item)
        {
           shared_region.elements[item] = item;
           shared_region.seqno.store(item + 1, std::memory_order_release);
           //asm("SFENCE"); // This seems to reproduce the performance regression in multicast ringbuffer on Alder Lake.
        }
        return std::pair{test_iters, 0UL};
    };
    auto read_func = [=](auto& shared_region,
                         std::size_t reader_index[[maybe_unused]],
                         std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto wasted_reads = 0UL;
        for(auto item=0UL; item != test_iters; ++item)
        {
            while(shared_region.seqno.load(std::memory_order_relaxed) == item)
            {
                asm("pause");
                ++wasted_reads;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if(shared_region.elements[item] != item) [[unlikely]] throw InvalidReadException("Read unexpected value. Thread race/bug.");
        }
        return std::pair{test_iters, wasted_reads};
    };
    analyze_result<std::size_t>(perf_test<decltype(construct_func),
                                decltype(emplace_func),
                                decltype(read_func),
                                readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<typename T,
         std::size_t capacity,
         std::size_t readers=1UL,
         std::size_t write_batch=1UL,
         std::size_t read_batch=1UL,
         bool        huge_pages = false>
void perf_test_qrius_blocking_ringbuff(qrius::CpuSet const& cpu_set,
                                       std::size_t test_iters) requires(std::is_constructible_v<T, std::size_t>)
{
    if constexpr (!readers) test_iters = std::min(test_iters, capacity); // Cannot write beyond 'capacity' elements if there are no readers.

    std::cout << "\nMultiCastRingBuff test_iterations="
              << test_iters
              << ", write_batch="
              << write_batch
              << ", read_batch="
              << read_batch
              << '\n';

    using MCRingBuff = qrius::RingBuff<T, !readers ? 1UL : readers, capacity, false>;
    auto construct_func = []()
    {
        auto ring_buff_ptr = make_helper<MCRingBuff, huge_pages>();
        assert(test_cacheline_align(ring_buff_ptr->get_writer()));
        assert(test_cacheline_align(ring_buff_ptr->get_reader(0UL)));
        return ring_buff_ptr;
    };
    auto emplace_func = [](auto& ring_buff,
                           std::size_t const test_iters) __attribute__((noinline, hot))
    {
        auto &writer = ring_buff.get_writer();
        writer.test_alignments();
        auto item = 0UL;
        auto wasted_ops = 0UL;
        if constexpr (write_batch == 1UL) // Unexplained HUGE improvement. TODO: Analyze!
        {
            for(; item != test_iters; ++item)
            {
                while(!writer.write(item))
                {
                    ++wasted_ops;
                    asm("pause");
                }
            }
        }
        else
        {
            while(item < test_iters)
            {
                auto slots = 0UL;
                while(!(slots = writer.template acquire<write_batch>()))
                {
                    ++wasted_ops;
                    asm("pause");
                }
                for(auto slot = 0UL; slot != slots; ++slot)
                {
                    writer.emplace(item++);
                }
                writer.commit();
            }
        }
        return std::pair{item, wasted_ops};
    };
    auto read_func = [](auto& ring_buff,
                        std::size_t reader_index,
                        std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto &reader = ring_buff.get_reader(reader_index);
        reader.test_alignments();
        auto item = 0UL;
        auto wasted_ops = 0UL;
        while(item < test_iters)
        {
            auto slots = 0UL;
            while(!(slots = reader.template acquire<read_batch>()))
            {
                ++wasted_ops;
                asm("pause");
            }
            for(auto slot=0UL; slot != slots; ++slot)
            {
                if(reader.read() != T{item})[[unlikely]] throw InvalidReadException("Unpexpected value read from ringbuff. Bug/Thread-race.");
                reader.pop();
                ++item;
            }
            reader.commit();
        }
        return std::pair{item, wasted_ops};
    };
    analyze_result<typename MCRingBuff::ValueType,
                   MCRingBuff::capacity,
                   readers+1>(perf_test<decltype(construct_func),
                                        decltype(emplace_func),
                                        decltype(read_func),
                                        readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<typename T, std::size_t capacity, std::size_t readers=1UL>
void perf_test_folly(qrius::CpuSet const& cpu_set, std::size_t test_iters)
{
    if constexpr (!readers) test_iters = capacity;
    std::cout << "\nfolly::ProducerConsumerQueue test_iterations=" << test_iters << '\n';
    using FollySPSCQueue = folly::ProducerConsumerQueue<T>;
    auto construct_func = []()
    {
        auto ring_buff_ptr = std::make_unique<FollySPSCQueue>(capacity + 1); // usable slots is size-1 (invariance of this queue)
        return ring_buff_ptr;
    };
    auto emplace_func = [](auto& ring_buff,
                           std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto item = 0UL;
        auto wasted_ops = 0UL;
        while(item < test_iters)
        {
            while(ring_buff.isFull())
            {
                ++wasted_ops;
                asm("pause");
            }
            auto result [[maybe_unused]] = ring_buff.write(item);
            assert(result);
            ++item;
        }
        return std::pair{item, wasted_ops};
    };
    auto read_func = [](auto& ring_buff,
                        std::size_t reader_index[[maybe_unused]],
                        std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto item = 0UL;
        auto wasted_ops = 0UL;
        while(item < test_iters)
        {
            while(ring_buff.isEmpty())
            {
                ++wasted_ops;
                asm("pause");
            }
            if(*ring_buff.frontPtr() != T{item}) [[unlikely]] throw InvalidReadException("folly Ringbuff read unexpected value");
            ring_buff.popFront();
            ++item;
        }
        return std::pair{item, wasted_ops};
    };
    analyze_result<typename FollySPSCQueue::value_type,
                   capacity,
                   readers+1>(perf_test<decltype(construct_func),
                                        decltype(emplace_func),
                                        decltype(read_func),
                                        readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

template<typename T, std::size_t capacity, std::size_t readers=1UL>
void perf_test_rigtorp(qrius::CpuSet const& cpu_set, std::size_t test_iters)
{
    if constexpr (!readers) test_iters = capacity;
    std::cout << "\n\nrigtorp::SPSCQueue, test_iterations=" << test_iters << '\n';
    using SPSCQueue = rigtorp::SPSCQueue<T>;
    auto construct_func = []()
    {
        return std::make_unique<SPSCQueue>(capacity+1);
    };
    auto emplace_func = [](auto& ring_buff,
                           std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto item = 0UL;
        auto wasted_ops = 0UL;
        for(item=0UL; item != test_iters; ++item)
        {
            while(!ring_buff.try_emplace(item))
            {
                ++wasted_ops;
                asm("pause");
            }
        }
        return std::pair{item, wasted_ops};
    };
    auto read_func = [](auto& ring_buff,
                        std::size_t reader_index[[maybe_unused]],
                        std::size_t test_iters) __attribute__((noinline, hot))
    {
        auto item = 0UL;
        auto wasted_ops = 0UL;
        while(item < test_iters)
        {
            T* value_ptr = nullptr;
            while(!(value_ptr = ring_buff.front()))
            {
                ++wasted_ops;
                asm("pause"); // yielding here improves writer throughput due to reader not contending on the cacheline (reducing l2_cache.rfo_miss. More details in my blog.)
            }
            if(*value_ptr != T{item}) [[unlikely]] throw InvalidReadException("rigtorp Ringbuff read unexpected value");
            ring_buff.pop();
            ++item;
        }
        return std::pair{item, wasted_ops};
    };
    analyze_result<T,
                   capacity,
                   readers+1>(perf_test<decltype(construct_func),
                                        decltype(emplace_func),
                                        decltype(read_func),
                                        readers>(cpu_set, construct_func, emplace_func, read_func, test_iters));
}

namespace argparse
{

enum class TestType
{
    all,
    invalid,
    local_store_load,
    shared_store_load,
    ping_pong,
    seqlock,
    seqlock_ringbuff,
    shared_region,
    shared_region_seqno,
    blocking_ringbuff,
    folly,
    rigtorp
};

using namespace std::string_view_literals;

TestType to_test_type(std::string_view test_type_arg) noexcept
{
    if(test_type_arg == ""sv)                       return TestType::all;
    if(test_type_arg == "local_store_load"sv)       return TestType::local_store_load;
    if(test_type_arg == "shared_store_load"sv)      return TestType::shared_store_load;
    if(test_type_arg == "ping_pong"sv)              return TestType::ping_pong;
    if(test_type_arg == "seqlock"sv)                return TestType::seqlock;
    if(test_type_arg == "seqlock_ringbuff"sv)       return TestType::seqlock_ringbuff;
    if(test_type_arg == "shared_region"sv)          return TestType::shared_region;
    if(test_type_arg == "shared_region_seqno"sv)    return TestType::shared_region_seqno;
    if(test_type_arg == "blocking_ringbuff"sv)      return TestType::blocking_ringbuff;
    if(test_type_arg == "folly"sv)                  return TestType::folly;
    if(test_type_arg == "rigtorp"sv)                return TestType::rigtorp;
    return TestType::invalid;
}

enum class StoredData
{
    small,
    aligned_small,
    large
};

StoredData to_stored_data_type(std::string_view stored_data_arg) noexcept
{
    if(stored_data_arg == "aligned"sv) return StoredData::aligned_small;
    if(stored_data_arg == "large"sv) return StoredData::large;
    return StoredData::small;
}

enum class Capacity
{
    small,
    medium,
    large
};

Capacity to_capacity(std::string_view capacity_arg) noexcept
{
    if(capacity_arg == "medium"sv) return Capacity::medium;
    if(capacity_arg == "large"sv)  return Capacity::large;
    return Capacity::small;
}

///
/// TODO: Clean up and simplify the whole argument parsing and various options
///
std::tuple<TestType, StoredData, Capacity, std::size_t> parse(int argc, char** argv) noexcept
{
    auto test_type_arg      = argc > 1 ? std::string_view(argv[1]) : ""sv;
    auto stored_data_arg    = argc > 2 ? std::string_view(argv[2]) : ""sv;
    auto capacity_arg       = argc > 3 ? std::string_view(argv[3]) : ""sv;
    auto readers            = argc > 4 ? std::min(argv[4][0] - '0', 5) : 1;
    return {to_test_type(test_type_arg),
            to_stored_data_type(stored_data_arg),
            to_capacity(capacity_arg),
            readers};
}

}

int main(int argc, char** argv)
{
    qrius::lock_pages();
    auto constexpr test_iters = 800'000'000UL;
    auto constexpr capacity = 1UL<<25;

    using namespace std::string_literals;
    using namespace argparse;
    auto cpu_set = qrius::CpuSet("011000"s);
    struct MarketData // 192 bytes
    {
        std::uint64_t id{};
        std::uint64_t seqno{};
        std::uint64_t volume{};
        std::uint64_t time_in_force{};
        std::uint64_t order_type{};
        std::uint64_t amend_id{};
        std::uint64_t best_bid{};
        std::uint64_t best_ask{};
        std::array<char, 128> symbol{};

        auto operator <=> (MarketData const& rhs) const = default;

        operator std::uint64_t() const noexcept
        {
            return id;
        }
    };

    struct CacheAlignedData
    {
        std::uint64_t data{};
        qrius::CachelinePadd<decltype(data)> padd{};

        auto operator == (CacheAlignedData const& rhs) const noexcept
        {
            return data == rhs.data;
        }

        operator std::uint64_t() const noexcept
        {
            return data;
        }
    };

    auto [test_type, data_type, capacity_type, readers] = argparse::parse(argc, argv);

    if(test_type == TestType::all || test_type == TestType::local_store_load)
    {
#if 0
        perf_test_local_store_load<std::uint64_t, 0>(cpu_set, 16'000'000'000UL);
#endif
        perf_test_local_store_load<std::uint64_t>(cpu_set, 16'000'000'000UL);
    }
    if(test_type == TestType::all || test_type == TestType::shared_store_load)
    {
#if 0
        perf_test_shared_store_load<std::uint64_t, 0>(cpu_set, 16'000'000'000UL);
#endif
        perf_test_shared_store_load<std::uint64_t>(cpu_set, 16'000'000'000UL);
    }
    if(test_type == TestType::all || test_type == TestType::ping_pong)
    {
        perf_test_ping_pong_store_load<std::uint64_t>(cpu_set, 40'000'000UL);
    }
    if(test_type == TestType::all || test_type == TestType::seqlock)
    {
#if 0
        perf_test_qrius_seqlock<std::uint64_t, 0UL>(cpu_set, 16'000'000'000UL); // readers = 0, no contention from reader
#endif
        perf_test_qrius_seqlock<std::uint64_t>(cpu_set, 16'000'000'000UL);
    }
    if(test_type == TestType::all || test_type == TestType::seqlock_ringbuff)
    {
        switch(data_type)
        {
            case StoredData::aligned_small:
                perf_test_seqlock_ringbuff<CacheAlignedData, capacity, 1>(cpu_set, test_iters);
                break;
            case StoredData::large:
                perf_test_seqlock_ringbuff<MarketData, capacity>(cpu_set, test_iters);
                break;
            default:
                switch(readers)
                {
                    case 0:
                        //
                        // 0 readers. This is the base benchmark.
                        // This is the highest possible throughput with no cacheline contention from
                        // reader 'spin-reading' the cacheline exclusive to the writer.
                        //
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity, 0>(cpu_set, test_iters);
                        break;
                    case 1:
                    default:
#if 0
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity, 1, true>(cpu_set, test_iters); // Test with ringbuffer allocated on Huge Pages
#endif
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity>(cpu_set, test_iters);
                        break;
                    case 2:
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity, 2>(qrius::CpuSet("111000"s), test_iters);
                        break;
                    case 3:
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity, 3>(qrius::CpuSet("111100"s), test_iters);
                        break;
                    case 4:
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity, 4>(qrius::CpuSet("111110"s), test_iters);
                        break;
                    case 5:
                        perf_test_seqlock_ringbuff<std::uint64_t, capacity, 5>(qrius::CpuSet("111111"s), test_iters);
                        break;
                }
        }
    }
    if(test_type == TestType::all || test_type == TestType::shared_region)
    {
       perf_test_shared_region<capacity, 1UL>(cpu_set, test_iters);
    }
    if(test_type == TestType::all || test_type == TestType::shared_region_seqno)
    {
       perf_test_shared_region_seqno<capacity, 1UL>(cpu_set, test_iters);
    }
    if(test_type == TestType::all || test_type == TestType::blocking_ringbuff)
    {
        switch(data_type)
        {
            case StoredData::aligned_small:
                perf_test_qrius_blocking_ringbuff<CacheAlignedData, capacity>(cpu_set, test_iters);
                break;
            case StoredData::large:
                perf_test_qrius_blocking_ringbuff<MarketData, capacity>(cpu_set, test_iters);
                break;
            default:
                switch(readers)
                {
                    case 0:
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 0>(cpu_set, test_iters);
                        break;
                    case 1:
                    default:
#if 0
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 1, 1, true>(cpu_set, test_iters); // Test with ringbuffer allocated on huge pages.
#endif
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity>(cpu_set, test_iters);
                        break;
                    case 2:
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 2>(qrius::CpuSet("111000"s), test_iters);
                        break;
                    case 3:
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 3>(qrius::CpuSet("111100"s), test_iters);
                        break;
                    case 4:
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 4>(qrius::CpuSet("111110"s), test_iters);
                        break;
                    case 5:
                        perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 5>(qrius::CpuSet("111111"s), test_iters);
                        break;
                }
#if 0
                //
                // 0 readers. This is the base benchmark.
                // This is the highest possible throughput with no cacheline contention from
                // reader spin reading the cacheline exclusive to the writer.
                //
                // These batching of writes (with max_slots acquired at a time set to  8-32) significantly
                // improves the overall throughput in tests.
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 0, 32>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 32, 1>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 2,  1>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 4,  1>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 8,  1>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 256,1>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 1,  2>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 1,  4>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 1,  256>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 1,  32>(cpu_set, test_iters);
                perf_test_qrius_blocking_ringbuff<std::uint64_t, capacity, 1, 32, 32>(cpu_set, test_iters);
#endif
                break;
        }
    }
    if(test_type == TestType::all || test_type == TestType::folly)
    {
        switch(data_type)
        {
            case StoredData::aligned_small:
                perf_test_folly<CacheAlignedData, capacity>(cpu_set, test_iters);
                break;
            case StoredData::large:
                perf_test_folly<MarketData, capacity>(cpu_set, test_iters);
                break;
            default:
                switch(readers)
                {
                    case 0:
                        //
                        // 0 readers. This is the base benchmark.
                        // This is the highest possible throughput with no cacheline contention from
                        // reader spin reading the cacheline exclusive to the writer.
                        //
                        perf_test_folly<std::uint64_t, capacity, 0UL>(cpu_set, test_iters);
                        break;
                    case 1:
                    default:
                        perf_test_folly<std::uint64_t, capacity>(cpu_set, test_iters);
                        break;
                }
        }
    }
    if(test_type == TestType::all || test_type == TestType::rigtorp)
    {
        switch(data_type)
        {
            case StoredData::aligned_small:
                perf_test_rigtorp<CacheAlignedData, capacity>(cpu_set, test_iters);
                break;
            case StoredData::large:
                perf_test_rigtorp<MarketData, capacity>(cpu_set, test_iters);
                break;
            default:
                switch(readers)
                {
                    case 0:
                        //
                        // 0 readers. This is the base benchmark.
                        // This is the highest possible throughput with no cacheline contention from
                        // reader spin reading the cacheline exclusive to the writer.
                        //
                        perf_test_rigtorp<std::uint64_t, capacity, 0UL>(cpu_set, test_iters);
                        break;
                    case 1:
                    default:
                        perf_test_rigtorp<std::uint64_t, capacity>(cpu_set, test_iters);
                    break;
                }
        }
    }
    return 0;
}