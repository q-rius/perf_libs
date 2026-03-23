#ifndef QRIUS_INCLUDE_GUARD_SEQLOCK_HPP
#define QRIUS_INCLUDE_GUARD_SEQLOCK_HPP

#include <type_traits>
#include <cassert>
#include <atomic>
#include <memory>

namespace qrius
{

///  
/// This is a Seqlock implementation used by multicast_ringbuff implementations.
/// APIs of which are designed for this specific usecase.
///
template<typename T> requires (std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>)
class Seqlock
{
public:
    using ValueType = T;
    using Seqno     = std::uint64_t;

    static_assert(std::atomic<Seqno>::is_always_lock_free && "can't support fast seqlock in this architecture");

    ///
    /// Writes the data with the associated seqno.
    ///
    /// Read will later provide the same associated seqno back.
    ///
    /// Seqno which is 64 bit must be monotonically increasing.
    ///
    /// Seqno is used in the context of the ringbuffer implementation that uses this.
    /// So it refers to the count of the latest item being written to the ringbuffer.
    ///
    /// Seqno overflow will result in undefined behavior for this implementation.
    /// This may not be as bad a limitation in practice, incrementing by 1 per cycle
    /// would need ~98 years to overflow on a 3GHZ processor.
    ///
    /// T must be trivially constructible from Args.
    ///
    void emplace(Seqno seqno, auto&&... args) noexcept requires (std::is_trivially_constructible_v<T, decltype(args)...>)
    {
        assert(data.version < seqno * 2 + 2 && "incoming seqno has to be monotonically increasing");
        data.version.store(seqno * 2 + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release); // This acts as a compiler barrier (is nop) on x86_64 (from TSO) but expensive on ARM (is required for correctness)
                                                             // A case where the extra silicon dedicated to load-store buffer unit proves its worth on x64_64.
        std::construct_at(&data.storage.value, std::forward<decltype(args)>(args)...);
        data.version.store(seqno*2 + 2, std::memory_order_release); // This is nop on x86_64 and very expensive on ARM - but is required for correctness.
    }

    ///
    /// Checks if the data available in the new slot is new enough
    /// when compared to the expected seqno.
    ///
    /// Must be true before attempting to read
    /// So this may be used as the polling API for the reader.
    ///
    bool read_ready(Seqno expected_seqno) const noexcept
    {
        auto requested_version = expected_seqno * 2 + 2;
        auto curr_version = data.version.load(std::memory_order_relaxed);
        return requested_version <= curr_version && !(curr_version & 1UL);
    }

    ///
    /// Returns a pair of the data and its corresponding
    /// seqno that was passed in during write.
    ///
    /// Must be read_ready() before calling this.
    /// Else can return data already read before.
    /// Staleness of the data can be determined from
    /// the returned seqno if read_ready was not used.
    ///
    /// Will still spin until an ongoing write is complete.
    /// seqno corresponding to T is also returned.
    ///
    /// T's copy constructor shouldn't throw which is a given from the
    /// already enforced stricter requirement that T must be trivially copyable.
    ///
    std::pair<T, Seqno> read() const noexcept
    {
        while(true)
        {
            auto const before = data.version.load(std::memory_order_acquire);
            if(before&1)
            {
                asm("pause");
                continue;
            }
            // Portability of this needs to be analyzed.
            // If another thread is simultaneously writing to data.storage.value,
            // it's undefined behavior. So compiler is free to hoist the below
            // copy outside the loop.
            // This messes up the algorithm. Here, we are hoping the thread_fence
            // should make the compiler realize there may be globally visible updates to the
            // memory out this function's context - like via a signal handler or something
            // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1478r1.html
            // It should still be good enough for our needs.
            auto result = data.storage.value;
            std::atomic_thread_fence(std::memory_order_acq_rel); // This acts as a compiler barrier (nop effectively) on x86_64 and is expensive on ARM - but is required for correctness.
                                                                 // It forces the result.store to stay above the fence and after.load to stay below the fence.
            auto const after = data.version.load(std::memory_order_relaxed);

            if(before == after && !(after & 1UL))   return {result, after/2 - 1};
        }
        unreachable();
    }

private:
    struct
    {
        std::atomic<Seqno> version{1UL}; // Can't start with zero since we want to wait on odd seqno.
        union Storage
        {
            std::byte a_byte;
            T value;
        } storage{std::byte{}};
    } data;
};

}
#endif