#ifndef QRIUS_INCLUDE_GUARD_PERF_TEST_HPP
#define QRIUS_INCLUDE_GUARD_PERF_TEST_HPP

#include <thread>
#include <future>
#include <barrier>
#include <iostream>
#include <chrono>

namespace qrius
{

struct TestResult
{
    using SteadyClockTP = std::chrono::time_point<std::chrono::steady_clock>;
    SteadyClockTP start_ts{};
    SteadyClockTP end_ts{};
    std::size_t   completed_ops{0};
    std::size_t   wasted_ops{0};
};

inline double throughput(TestResult const& result) noexcept
{
    assert(result.end_ts >= result.start_ts && "bad clock or timestamps");
    using namespace std::chrono_literals;
    return static_cast<double>(result.completed_ops)/((result.end_ts - result.start_ts)/1ns)*1'000'000'000UL;
}

inline double latency(TestResult const& result) noexcept
{
    assert(result.end_ts >= result.start_ts && "bad clock or timestamps");
    using namespace std::chrono_literals;
    return ((result.end_ts - result.start_ts)/1ns)/static_cast<double>(result.completed_ops);
}

///
/// @brief 
/// @tparam ConstructFunc Constructs the data structure being performance tested in the writer thread (numa aware allocation).
/// @tparam EmplaceFunc Creates/Writes all test data to the shared data structure being performance tested.
/// @tparam ReadFunc Reads all data being written to by writer thread - invoked in each reader threads
/// @tparam reader_count Number of readers/reader threads to be created/tested.
/// @param cpu_set cpu set to which writer + reader threads need to be glued to (affinity).
/// @param construct_func
/// @param emplace_func 
/// @param read_func 
/// @param test_iters Number of items to produced and read. 
/// @return Returns the final throughput details of each test.
///
/// Reader/s cursor/s must reach the last element produced by writer for the tests to finish.
/// Throughput of the writer, reader, both combined and wasted operations are reported for each test.
///

template<std::invocable<> ConstructFunc,
         typename EmplaceFunc,
         typename ReadFunc,
         std::size_t reader_count=1,
         std::size_t stack_prefault=(1UL<<16)>
std::array<qrius::TestResult, reader_count+1>
    perf_test(qrius::CpuSet const& cpu_set,
              ConstructFunc&& construct_func,
              EmplaceFunc&& emplace_func,
              ReadFunc&& read_func,
              std::size_t test_iters)
{
    using namespace std::chrono_literals;

    std::array<qrius::TestResult, reader_count+1> result;
    {
        using Type = typename std::decay_t<std::invoke_result_t<ConstructFunc>>::element_type; // ConstructFunc must return a unique_ptr to shared data structure being throughput tested
        std::vector<std::promise<Type&>> promises(reader_count);
        std::vector<std::future<Type&>> futures;
        for(auto& prom : promises) futures.emplace_back(prom.get_future());

        std::barrier start_barrier(reader_count + 1);
        std::barrier end_barrier(reader_count + 1);
        std::jthread producer(
            [&]()
            {
                qrius::force_page_fault_stack<stack_prefault>();
                auto core_id = qrius::nth_set_bit(cpu_set, 1UL);
#ifndef NDEBUG
                std::cout << "starting writer   target = "
                          << test_iters
                          << ", core_affinity = "
                          << core_id
                          << '\n';
#endif
                if(!qrius::set_curr_thread_affinity(core_id))
                {
                    std::cerr << "Failed to set affinity of the producer thread to the core " << core_id << '\n';
                }
                // Construct the data structure being tested in the writer thread for numa friendly allocaiton.
                // Instead of the thread that sets up the test.
                auto data_ptr = construct_func();
                auto &data = *data_ptr;

                // Hand over the data structure being tested to all readers.
                for(auto& prom : promises)
                {
                    prom.set_value(data);
                }

                start_barrier.arrive_and_wait(); // May not be exception safe - might indefinitely wait if one of the threads throw or exit unexpectedly

                std::atomic_thread_fence(std::memory_order_seq_cst); // Above barrier might be enough.
                auto writer_start_ts = std::chrono::steady_clock::now();

                std::atomic_signal_fence(std::memory_order_seq_cst); // make sure compiler doesn't reorder
                auto [write_count, wasted_ops] = emplace_func(data, test_iters);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                auto writer_end_ts = std::chrono::steady_clock::now();
                std::atomic_thread_fence(std::memory_order_seq_cst); // Serialize instructions

                result[0] = {writer_start_ts, writer_end_ts, write_count, wasted_ops};
                end_barrier.arrive_and_wait(); // Probably not exception safe
            }
        );
        std::array<std::jthread, reader_count> reader_threads;
        auto reader_index [[maybe_unused]]= 0UL;
        for(auto &reader_thread : reader_threads)
        {
            reader_thread = std::jthread(
                [&, reader_index]()
                {
                    qrius::force_page_fault_stack<stack_prefault>();
                    auto core_id = qrius::nth_set_bit(cpu_set, reader_index + 2); // producer takes the first set cpu
#ifndef NDEBUG
                    std::cout << "starting reader "
                              << reader_index
                              << " target = "
                              << test_iters
                              << ", core_affinity = "
                              << core_id
                              << '\n';
#endif
                    if(!qrius::set_curr_thread_affinity(core_id))
                    {
                        std::cerr << "Failed to set affinity of the reader thread "
                                  << reader_index
                                  << " to the core "
                                  << core_id
                                  << '\n';
                    }
                    futures[reader_index].wait(); // May be timeout for exception safety from other threads exiting quickly
                    auto& data = futures[reader_index].get();
                    start_barrier.arrive_and_wait();

                    std::atomic_thread_fence(std::memory_order_seq_cst); // Above barrier might be enough though but doesn't hurt.
                    auto reader_start_ts = std::chrono::steady_clock::now(); // May need a fence/barrier here to prevent reordering. X86_64 assembly looks ok

                    std::atomic_signal_fence(std::memory_order_seq_cst); // So that compiler doesn't reorder.
                    auto [read_count, wasted_ops] = read_func(data, reader_index, test_iters);
                    std::atomic_signal_fence(std::memory_order_seq_cst);

                    auto reader_end_ts = std::chrono::steady_clock::now();
                    std::atomic_thread_fence(std::memory_order_seq_cst);

                    result[reader_index+1] = {reader_start_ts, reader_end_ts, read_count, wasted_ops};
                    end_barrier.arrive_and_wait();
                }
            );
            ++reader_index;
        }
    }
    return result;
}

//
/// Tagged as cold so as to not interfere/inline/optimize with the code being
/// performance tested.
/// Intention is to obtain an apples to apples comparison of the actual code
/// being performance tested.
/// Especially under aggressive optimization levels (O3) in gcc/clang
///
template<typename value_type, std::size_t queue_capacity=1UL, std::size_t size=2UL>
static void analyze_result(std::array<qrius::TestResult, size> const& result) __attribute__((cold));

template<typename value_type, std::size_t queue_capacity, std::size_t size>
static void analyze_result(std::array<qrius::TestResult, size> const& result)
{
    constexpr auto reader_count = size - 1; // Assumption is single writer multiple readers
    std::cout << "capacity="
              << queue_capacity
              << " elements, readers="
              <<  reader_count
              << " threads, element_size="
              << sizeof(value_type)
              << " bytes\n";

    using namespace std::chrono_literals;
    auto const& writer = result[0];
    auto writer_mps = qrius::throughput(writer);
    std::cout << "writer produced "
            << writer.completed_ops
            << " in "
            << (writer.end_ts - writer.start_ts)/1ns
            << " ns, wasted_writes= "
            << writer.wasted_ops
            << '\n';
    std::cout << "  throughput="
            << writer_mps
            << " msgs/sec, "
            << writer_mps * sizeof(value_type) / (1UL << 30)
            << " GBPS, "
            << writer_mps * sizeof(value_type) * 8 / 1'000'000'000UL
            << " Gbps, avg_latency="
            << latency(writer)
            << "ns\n";

    auto last_to_complete = result[0];
    for(auto i=1UL; i!=size; ++i)
    {
        auto const& reader = result[i];
        std::cout << "reader "
                    << i-1
                    << " read "
                    << reader.completed_ops
                    << " in "
                    << (reader.end_ts - reader.start_ts)/1ns
                    << "ns, wasted_reads= "
                    << reader.wasted_ops
                    << '\n';
        auto mps = qrius::throughput(reader);
        std::cout << "  throughput="
                  << mps
                  << " msgs/sec, "
                  << mps * sizeof(value_type) / (1UL << 30)
                  << " GBPS, "
                  << mps * sizeof(value_type) * 8 / 1'000'000'000UL
                  << " Gbps, avg_latency="
                  << latency(reader)
                  <<"ns\n";
        std::cout << "  started "
                  << (reader.start_ts - writer.start_ts)/1ns
                  << "ns after writer\n";
        if(last_to_complete.end_ts < reader.end_ts)    last_to_complete = reader;
    }
    auto effective_mps = static_cast<double>(last_to_complete.completed_ops)/((last_to_complete.end_ts - writer.start_ts)/1ns) * 1'000'000'000UL;
    std::cout << "effective system throughput="
              << effective_mps
              << " msg/sec, "
              << effective_mps * sizeof(value_type) / (1UL << 30)
              << " GBPS, "
              << effective_mps * sizeof(value_type) * 8 / 1'000'000'000UL
              << " Gbps, avg_latency="
              << 1'000'000'000UL/effective_mps
              << "ns (influenced by latency to start reader threads & scheduling etc.)\n";
}

}
#endif