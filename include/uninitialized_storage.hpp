#ifndef QRIUS_INCLUDE_GUARD_UNINITIALIZED_STORAGE_HPP
#define QRIUS_INCLUDE_GUARD_UNINITIALIZED_STORAGE_HPP

#include <perf_utils.hpp>

#include <cstddef>
#include <type_traits>
#include <memory>
#include <array>

namespace qrius
{

///
/// This is a helper to hold the underlying storage of the ringbuffer implementations.
/// This makes the ringbuffer value types.
/// This is not a generic storage implementation in the sense, it
/// doesn't manage the lifetime of the objects in the storage.
/// Objects have to be constructed and destroyed using the relevant APIs provided here
/// by the user i.e. when UninitializedArray goes out of scope, it doesn't invoke
//  the destructors of any objects that may be constructed in the storage.
/// For the use cases this is intended for, there is specific lifetime management
/// required for each object.
///
template<typename T, std::size_t size, bool force_page_fault_at_init=true> requires (size > 0UL)
class UninitializedArray
{
public:
    UninitializedArray() requires (!force_page_fault_at_init) = default;

    constexpr UninitializedArray() noexcept
    {
        static_assert(force_page_fault_at_init); // so we are ok to compromise on constexpr-ness.
        if constexpr (can_be_array)
        {
            force_page_fault(reinterpret_cast<std::byte*>(&storage[0]), sizeof(storage));
        }
        else
        {
            force_page_fault(storage, sizeof(storage));
        }
    }

    constexpr T& operator [] (std::size_t at) noexcept
    {
        if constexpr (can_be_array)
        {
            return storage[at];
        }
        else
        {
            return *std::launder(reinterpret_cast<T*>(&storage[at*sizeof(T)]));
        }
    }

    constexpr T const& operator [] (std::size_t at) const noexcept
    {
        if constexpr (can_be_array)
        {
            return storage[at];
        }
        else
        {
            return *std::launder(reinterpret_cast<T const*>(&storage[at*sizeof(T)]));
        }
    }

    constexpr void construct_at(std::size_t at, auto&&... args) noexcept(std::is_nothrow_constructible_v<T, decltype(args)...>)
    {
        if constexpr (can_be_array)
        {
            std::construct_at(&storage[at], std::forward<decltype(args)>(args)...);
        }
        else
        {
            std::construct_at(std::launder(reinterpret_cast<T*>(&storage[at*sizeof(T)])), std::forward<decltype(args)>(args)...);
        }
    }

    constexpr void destroy_at(std::size_t at) noexcept
    {
        if constexpr (!can_be_array)
        {
            std::destroy_at(std::launder(reinterpret_cast<T*>(&storage[at*sizeof(T)])));
        }
    }
private:
    constexpr static auto can_be_array = std::is_trivially_constructible_v<T> && std::is_trivially_destructible_v<T>;
    using ByteStorage = std::byte[sizeof(T)*size];
    using StorageType = std::conditional_t<can_be_array,
                                           std::array<T, size>,
                                           ByteStorage>;
    alignas(alignof(T)) StorageType storage;
};

}

#endif