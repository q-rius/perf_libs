#ifndef QRIUS_INCLUDE_GUARD_PERF_UTILS_HPP
#define QRIUS_INCLUDE_GUARD_PERF_UTILS_HPP

#include <atomic>
#include <cassert>
#include <bitset>
#include <chrono>
#include <bit>
#include <iostream>
#include <array>

#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>
#include <immintrin.h>

namespace qrius
{

constexpr inline auto cacheline_size = 64UL; // = std::hardware_destructive_interference_size
constexpr inline auto page_size = 4096UL;

template<typename T>
inline constexpr std::size_t cacheline_align = std::max(cacheline_size, alignof(T));

///
/// Helper for cacheline_padding in bytes for a series of fields that
/// needs to be grouped together in cachelines inside a struct or class.
/// struct Foo
/// {
///     int a;
///     int& b;
///     char c;
///     std::byte padding[instead_of_calculating_yourself];
/// };
///
/// use
///
/// struct Foo
/// {
///     int  a;
///     int& b;
///     char c;
///     CachelinePadd<int, int&, char> padd{}; // a, b, c will stay in the cacheline assuming Foo is at least aligned to cacheline.
/// };
/// Assumes std::tuple generates the same memory layout as a struct with members defined in the same level (non-nested).
///

template<typename... T>
inline constexpr auto cacheline_padding = sizeof(std::tuple<T...>) % cacheline_size
                                                ? cacheline_size - (sizeof(std::tuple<T...>) % cacheline_size)
                                                : 0UL;
template<typename T>
using CachelinePadd = std::byte[cacheline_padding<T>];

[[noreturn]] inline void unreachable()
{
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#ifdef __GNUC__ // GCC, Clang, ICC
    __builtin_unreachable();
#else
    static_assert(false && "unsupported compiler platform");
#endif
}

inline constexpr bool test_alignment(auto const& value, std::size_t alignment) noexcept
{
    return std::bit_cast<const std::uintptr_t>(std::addressof(value)) % alignment == 0UL;
}

inline constexpr bool test_cacheline_align(auto const& value) noexcept
{
    return std::bit_cast<const std::uintptr_t>(std::addressof(value)) % cacheline_size == 0UL;
}

constexpr inline auto max_cpus = 128UL;
using CpuSet = std::bitset<max_cpus>;

inline bool set_curr_thread_affinity(std::size_t core_id) noexcept
{
    assert(core_id < CPU_SETSIZE);
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(core_id, &cpu_set);
    return !sched_setaffinity(gettid(), core_id + 1, &cpu_set);
}

///
/// Returns the index of the nth set bit in bitset.
/// if not found, returns size of the bitset representing
/// an out of bound index.
///
template<std::size_t size>
inline constexpr std::size_t nth_set_bit(std::bitset<size> bset, std::size_t n) noexcept
{
    assert(n !=0 && n<=size);
    auto pos = 0UL;
    for(auto i=0UL; i!=size; ++i)
    {
        pos += bset[i];
        if(pos == n) return i;
    }
    return size;
}

inline void lock_pages() noexcept
{
    //
    // Note this produces a significant throughput increase
    // for buffer in the heap on my host (ubuntu).
    // perf shows significant reduction in the page_faults stat.
    // mlockall is not good idea in a real app.
    // But it does the job for the performance test in context.
    // Ideally, one should walk the address space and specifically lock
    // the required pages via /proc/$pid/maps on Linux.
    //
    auto rc [[maybe_unused]]= mlockall(MCL_CURRENT|MCL_FUTURE);
#ifndef NDEBUG
    std::cout << "mlockall rc=" << rc << ", errno=" << (rc==0 ? 0 : errno) << ", errstr="  << (rc == 0 ? "NA" : strerror(errno)) << "\n";
#endif
}

template<typename T>
inline void clflush(T const& obj) noexcept
{
    auto const begin = std::bit_cast<std::byte const*>(&obj);
    auto const end = begin + sizeof(T);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for(auto iter=begin; iter<end; iter += cacheline_size)
    {
        _mm_clflush(iter);
    }
    _mm_clflush(end);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

// TODO: Translate API to a continguous_range concept
template<typename T>
inline void clflush_region(T const* region, std::size_t size) noexcept
{
    auto const begin = std::bit_cast<std::byte const*>(region);
    auto const end = begin + size * sizeof(T);
    std::atomic_thread_fence(std::memory_order_seq_cst); // serialize clflush-es : to defend against compiler reordering
    for(auto iter=begin; iter<end; iter += cacheline_size)
    {
        _mm_clflush(iter);
    }
    _mm_clflush(end);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

inline void force_page_fault(std::byte* region, std::size_t size) noexcept
{
    assert(size>0);
    for(auto byte=region; byte < region+size; byte += page_size)
    {
        *byte = std::byte{'\0'};
    }
    *(region+size-1) = std::byte{'\0'};
}

template<std::size_t bytes>
inline void force_page_fault_stack() noexcept
{
    std::byte array[bytes];
    asm volatile("" ::"m"(array) :"memory"); // Like DoNotOptimize in MicroBenches.
    force_page_fault(array, bytes);
}


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

}
#endif