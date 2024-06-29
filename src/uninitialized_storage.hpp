#ifndef QRIUS_INCLUDE_GUARD_UNINITIALIZED_STORAGE_HPP
#define QRIUS_INCLUDE_GUARD_UNINITIALIZED_STORAGE_HPP

#include <cstddef>
#include <type_traits>
#include <memory>

namespace qrius
{

///
/// This is a helper to hold the underlying storage of the ringbuffer implementations.
/// This makes the ringbuffer value types.
/// This is not a generic storage implementation in the sense, it
/// doesn't manage the lifetime of the objects in the storage.
/// Objects have to be constructed and destroyed using the relevant APIs provided here
/// by the user i.e. when UninitializedStorage goes out of scope, it doesn't invoke
//  the destructors of any objects that may be constructed in the storage.
/// For the use cases this is intended for, there is specific lifetime management
/// required for each object.
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
    T& operator [] (std::size_t at) noexcept
    {
        return *std::launder(reinterpret_cast<T*>(&storage[at*sizeof(T)]));
    }

    T const& operator [] (std::size_t at) const noexcept
    {
        return *std::launder(reinterpret_cast<T const*>(&storage[at*sizeof(T)]));
    }

    void construct_at(std::size_t at, auto&&... args) noexcept(std::is_nothrow_constructible_v<T, decltype(args)...>)
    {
        std::construct_at(std::launder(reinterpret_cast<T*>(&storage[at*sizeof(T)])), std::forward<decltype(args)>(args)...);
    }

    void destroy_at(std::size_t at) noexcept
    {
        std::destroy_at(std::launder(reinterpret_cast<T*>(&storage[at*sizeof(T)])));
    }
private:
    alignas(alignof(T)) std::byte storage[sizeof(T)*size];
};

}

#endif