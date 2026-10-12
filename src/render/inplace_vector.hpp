#pragma once

// Fixed-capacity, stack-resident vector (C++26 P0843 std::inplace_vector).
// MSVC 19.51's STL does not ship <inplace_vector> yet, so this header falls back to a
// minimal implementation covering the subset OpenF4 uses. Once the library provides it,
// the alias switches to the standard type with no call-site changes.

#include <cstddef>
#include <version>

#if defined(__cpp_lib_inplace_vector)
#include <inplace_vector>
namespace openf4
{
    template <typename T, std::size_t Capacity>
    using inplace_vector = std::inplace_vector<T, Capacity>;
}
#else
#include <array>
#include <cassert>
#include <type_traits>

namespace openf4
{
    // Restricted to trivially copyable element types: no per-element lifetime management,
    // which is all the bounded scratch buffers in hot paths need.
    template <typename T, std::size_t Capacity>
    class inplace_vector
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>,
            "openf4::inplace_vector fallback only supports trivial element types");

    public:
        using value_type = T;
        using size_type = std::size_t;
        using iterator = T*;
        using const_iterator = const T*;

        constexpr inplace_vector() noexcept = default;

        static constexpr size_type capacity() noexcept { return Capacity; }
        static constexpr size_type max_size() noexcept { return Capacity; }
        [[nodiscard]] constexpr size_type size() const noexcept { return mSize; }
        [[nodiscard]] constexpr bool empty() const noexcept { return mSize == 0; }

        constexpr T* data() noexcept { return mStorage.data(); }
        constexpr const T* data() const noexcept { return mStorage.data(); }
        constexpr iterator begin() noexcept { return data(); }
        constexpr iterator end() noexcept { return data() + mSize; }
        constexpr const_iterator begin() const noexcept { return data(); }
        constexpr const_iterator end() const noexcept { return data() + mSize; }
        constexpr T& operator[](size_type index) noexcept { assert(index < mSize); return mStorage[index]; }
        constexpr const T& operator[](size_type index) const noexcept { assert(index < mSize); return mStorage[index]; }
        constexpr T& front() noexcept { assert(mSize); return mStorage[0]; }
        constexpr const T& front() const noexcept { assert(mSize); return mStorage[0]; }
        constexpr T& back() noexcept { assert(mSize); return mStorage[mSize - 1]; }
        constexpr const T& back() const noexcept { assert(mSize); return mStorage[mSize - 1]; }

        // Like std::inplace_vector::push_back, exceeding the capacity is a precondition
        // violation; callers flush before that point.
        constexpr T& push_back(const T& value) noexcept
        {
            assert(mSize < Capacity);
            mStorage[mSize] = value;
            return mStorage[mSize++];
        }
        constexpr void clear() noexcept { mSize = 0; }

    private:
        std::array<T, Capacity> mStorage{};
        size_type mSize = 0;
    };
}
#endif
