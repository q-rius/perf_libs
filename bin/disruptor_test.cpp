// TODO: Convert to GTest.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <perf_utils.hpp>
#include <uninitialized_storage.hpp>

#include <type_traits>
#include <array>
#include <algorithm>

namespace qrius
{

template<typename T, std::size_t alignment=alignof(T)>
class alignas(alignment) UninitializedStorage
{
  public:

    template<typename... Args>
    constexpr void construct(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
    {
        if constexpr (can_be_t)
        {
            std::construct_at(&storage, std::forward<Args>(args)...); /// This must be noexcept for this to be exception safe.
                                                               /// Since each element of ringbuffer is specifically controlled/constructed by
                                                               /// tracking seqno, this may be OK - TODO.
        }
        else
        {
           std::construct_at(reinterpret_cast<T*>(storage), std::forward<Args>(args)...);
        }
    }
    /// 
    /// Assumes T is in constructed state
    /// i.e. construct has been invoked but not
    /// destroyed yet.
    /// 
    constexpr T const& get() const& noexcept
    {
        if constexpr (can_be_t)
        {
            return storage;
        }
        else
        {
            return *std::launder(reinterpret_cast<T const*>(storage));
        }
    }

    /// 
    /// Assumes T is in constructed state
    /// 
    constexpr T& get() & noexcept
    {
        if constexpr (can_be_t)
        {
            return storage;
        }
        else
        {
            return *std::launder(reinterpret_cast<T*>(storage));
        }
    }

    /// 
    /// Assumes T is in constructed state
    /// 
    constexpr T&& get() && noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        if constexpr (can_be_t)
        {
            return std::move(storage);
        }
        else
        {
            return std::move(*std::launder(reinterpret_cast<T*>(storage)));
        }
    }

    constexpr void destroy() noexcept
    {
        if constexpr(!can_be_t)
        {
            std::destroy_at(&storage);
        }
    }

  private:
    static constexpr bool can_be_t = std::is_trivially_constructible_v<T> &&
                                     std::is_trivially_destructible_v<T>;
    using ByteStorage = std::byte[sizeof(T)];
    using StorageT = std::conditional_t<can_be_t,
                                        T,
                                        ByteStorage>;
    StorageT storage;
};

template<typename... Ts>
class RingElement
{
  public:
    ///
    /// An element can be constructed once before RingElement is destroyed.
    /// Works for ring buffer like scenarios where element is constructed and destroyed
    /// before being re-used.
    ///
    template<typename U, typename... Args>
    constexpr void construct(Args&&... args) noexcept(std::is_nothrow_constructible_v<U, Args...>)
    {
        using StorageT = UninitializedStorage<U, cacheline_align<U>>;
        std::get<StorageT>(values).construct(std::forward<Args>(args)...);
    }

    template<typename U>
    constexpr void destroy() noexcept
    {
        using StorageT = UninitializedStorage<U, cacheline_align<U>>;
        std::get<StorageT>(values).destroy();
    }

    template<typename U>
    constexpr U& get() & noexcept
    {
        using StorageT = UninitializedStorage<U, cacheline_align<U>>;
        return std::get<StorageT>(values).get();
    }

    template<typename U>
    constexpr U const& get() const& noexcept
    {
        using StorageT = UninitializedStorage<U, cacheline_align<U>>;
        return std::get<StorageT>(values).get();
    }

    template<typename U>
    constexpr U&& get() && noexcept(std::is_nothrow_move_constructible_v<U>)
    {
        using StorageT = UninitializedStorage<U, cacheline_align<U>>;
        return std::get<StorageT>(values).get();
    }

  private:

    std::tuple<UninitializedStorage<Ts, cacheline_align<Ts>>...> values;
};

template<typename T>
class RingElement<T> : public UninitializedStorage<T>
{
};

using Seqno = std::size_t;

template<std::size_t capacity, typename... Elems> requires (!(capacity & (capacity-1)))
class RingBuffer
{
  public:
    template<typename U, typename... Args>
    constexpr void construct_at(Seqno seqno, Args&&... args) noexcept(std::is_nothrow_constructible_v<U, Args...>)
    {
        ring_buffer[index(seqno)].template construct<U>(std::forward<Args>(args)...);
    }

    template<typename U>
    constexpr void destroy_at(Seqno seqno) noexcept
    {
        ring_buffer[index(seqno)].template destroy<U>();
    }

    template<typename U>
    constexpr U& get(Seqno seqno) & noexcept
    {
        return ring_buffer[index(seqno)]. template get<U>();
    }

    template<typename U>
    constexpr U const& get(Seqno seqno) const& noexcept
    {
        return ring_buffer[index(seqno)]. template get<U>();
    }

    template<typename U>
    constexpr U&& get(Seqno seqno) && noexcept(std::is_nothrow_move_constructible_v<U>)
    {
        return ring_buffer[index(seqno)]. template get<U>();
    }

  private:

    static constexpr auto index(Seqno seqno) noexcept
    {
        return seqno & (capacity-1);
    }

    std::array<RingElement<Elems...>, capacity> ring_buffer;
};

template<std::size_t capacity, typename Elem> requires (!(capacity & (capacity-1)))
class RingBuffer<capacity, Elem>
{
  public:
    template<typename... Args>
    constexpr void construct_at(Seqno seqno, Args&&... args) noexcept(std::is_nothrow_constructible_v<Elem, Args...>)
    {
        ring_buffer[index(seqno)].construct(std::forward<Args>(args)...);
    }

    constexpr void destroy_at(Seqno seqno) noexcept
    {
        ring_buffer[index(seqno)].destroy();
    }

    constexpr Elem& get(Seqno seqno) & noexcept
    {
        return ring_buffer[index(seqno)].get();
    }

    constexpr Elem const& get(Seqno seqno) const& noexcept
    {
        return ring_buffer[index(seqno)].get();
    }

    constexpr Elem&& get(Seqno seqno) && noexcept(std::is_nothrow_move_constructible_v<Elem>)
    {
        return ring_buffer[index(seqno)].get();
    }

  private:

    static constexpr auto index(Seqno seqno) noexcept
    {
        return seqno & (capacity-1);
    }

    std::array<RingElement<Elem>, capacity> ring_buffer;
};

class Barrier
{
  public:
    Seqno snoop_seqno() const noexcept
    {
        return committed_seqno.load(std::memory_order_acquire);
    }

    Seqno seqno() const noexcept
    {
        return committed_seqno.load(std::memory_order_relaxed);
    }

    void update_seqno(Seqno seqno) noexcept
    {
        committed_seqno.store(seqno, std::memory_order_release);
    }
  private:
    std::atomic<Seqno> committed_seqno{0};
};

namespace dag
{

template<std::size_t level, std::size_t next_stages>
class Stage
{
    static constexpr auto level = level;
    static constexpr auto next_stages = next_stages;
    static constexpr bool last_stage = false;
};

template<std::size_t level>
class Stage<level, 0>
{
    static constexpr auto level = level;
    static constexpr auto next_stages = 0UL;
    static constexpr bool last_stage = true;
};

}

template<std::size_t capacity, std::size_t final_stage_count, typename... Elems> requires(final_stage_count != 0UL)
class Disruptor
{
  public:
    using RingBufferT = RingBuffer<capacity, Elems...>;

    template<std::size_t deps>
    class Producer
    {
      public:
        Producer(Disruptor& disruptor)
            : ring_buffer(disruptor.ring_buffer)
        {}

        void set_barrier(std::array<Barrier*, deps> const& barriers)
        {
            consumer_barriers = barriers;
        }

        template<typename Elem, typename... Args>
        void write(Args&&... args) noexcept(std::is_nothrow_constructible_v<Elem, Args...>)
        {
            assert(consumer_barriers && "consumer that last commits to ring must be known to the writer");
            auto seqno = barrier.seqno();
            assert(cached_consumer_seqno + capacity <= seqno);
            while(cached_consumer_seqno + capacity == seqno)
            {
                cached_consumer_seqno = std::ranges::min_element(consumer_barriers,
                                                                 [](auto a, auto b)
                                                                 {
                                                                    return a->snoop_seqno() < b->snoop_seqno();
                                                                 });
            }
            ring_buffer.construct_at(seqno, std::forward<Args>(args)...);
            barrier.update_seqno(seqno+1);
        }

      private:
        RingBufferT& ring_buffer;
        std::array<Barrier*, final_stage_count>   consumer_barriers;
        Seqno     cached_consumer_seqno{0};
        alignas(cacheline_size) Barrier barrier;
        CachelinePadd<Barrier>  padd;
    };

    template<std::size_t deps=1, bool final_stage=1>
    class Consumer
    {
      public:
       Consumer(Disruptor& disruptor)
            : ring_buffer(disruptor.ring_buffer)
        {}
    
        void set_dependencies(std::array<Barrier*, deps> const& dep_barriers) noexcept
        {
            dependent_barriers = dep_barriers;
        }

        std::size_t acquire() noexcept 
        {
            assert(cached_dependent_seqno >= in_progress_seqno);
            auto const seqno = in_progress_seqno;
            while(cached_dependent_seqno == seqno)
            {
                cached_dependent_seqno = std::ranges::min_element(dependent_barriers,
                                                                  [](auto a, auto b)
                                                                  {
                                                                    return a->snoop_seqno() < b->snoop_seqno();
                                                                  });

            }
            assert(cached_dependent_seqno > seqno);
            return cached_dependent_seqno - seqno;
        }

        template<typename T>
        T const& read() const noexcept
        {
            return ring_buffer. template get<T>(in_progress_seqno);
        }

        template<typename T>
        T& read() & noexcept
        {
            return ring_buffer. template get<T>(in_progress_seqno);
        }

        template<typename T>
        T&& read() && noexcept(std::is_nothrow_move_constructible_v<T>)
        {
            return ring_buffer. template get<T>(in_progress_seqno);
        }

        void mark() noexcept requires(!final_stage || final_stage_count>1)
        {
            ++in_progress_seqno;
        }

        void pop() noexcept requires(final_stage && final_stage_count==1)
        {
            if constexpr(final_stage && final_stage_count==1)
            {
                ring_buffer.destroy_at(in_progress_seqno);
            }
            ++in_progress_seqno;
        }

        void commit() noexcept
        {
            barrier.update_seqno(in_progress_seqno);
        }

        ~Consumer() requires (final_stage && final_stage_count==1)
        {
            clear();
        }
      private:
        void snoop_seqno() noexcept
        {
            cached_dependent_seqno = std::ranges::min_element(dependent_barriers,
                                                              [](auto a, auto b)
                                                              {
                                                                    return a->snoop_seqno() < b->snoop_seqno();
                                                              });

        }
        void clear() requires (final_stage && final_stage_count==1)
        {
            snoop_seqno();
            for(auto seqno=in_progress_seqno; seqno <= cached_dependent_seqno; ++seqno)
            {
                ring_buffer.destroy_at(seqno);
            }
        }

        RingBufferT&                ring_buffer;
        Seqno                       cached_dependent_seqno{0};
        Seqno                       in_progress_seqno{0};
        std::array<Barrier*, deps>  dependent_barriers;
        alignas(cacheline_size) Barrier      barrier;
        CachelinePadd<Barrier>  padd;
    };
  private:
    RingBuffer<capacity, Elems...> ring_buffer;
};

}

static void uninit_storage_test() noexcept
{
    static_assert(alignof(qrius::UninitializedStorage<char>) == alignof(char));
    static_assert(alignof(qrius::UninitializedStorage<char, qrius::cacheline_size>) == qrius::cacheline_size);
    {
        struct TestData
        {
            TestData() = delete;
            TestData(int a)
                : a{a}
            {}
            bool operator == (TestData const&) const = default;
            int a;
            int b{};
            int c{};
        };

        qrius::UninitializedStorage<TestData> test;
        test.construct(37);    
        assert(test.get() == TestData{37});
        test.destroy();
    }
}

static void ringelement_test() noexcept
{
    {
        qrius::RingElement<int> ring_element;
        static_assert(sizeof(ring_element) == sizeof(int));
        ring_element.construct(37);
        assert(ring_element.get() == 37);
    }
    {
        struct TestData
        {
            int a;
            int b;
            bool operator == (TestData const&) const = default;
        };
        qrius::RingElement<TestData> ring_element;
        ring_element.construct(10, 11);
        assert((ring_element.get() == TestData{10, 11}));
    }
    {
        struct TestData
        {
            int a{};
            char b;
            TestData() = delete;
            TestData(char b)
                : b{b}
            {}
            bool operator == (TestData const&) const = default;
        };
        qrius::RingElement<TestData> ring_element;
        ring_element.construct('b');
        assert((ring_element.get() == TestData{'b'}));
    }
    {
        qrius::RingElement<int, float, double, std::size_t> ring_element;
        static_assert(sizeof(ring_element) == qrius::cacheline_size*4);
        ring_element.construct<int>(100);
        assert(ring_element.get<int>() == 100);
        ring_element.construct<float>(0.0);
        assert(ring_element.get<float>() == 0.0);
    }
    {
        struct TestData1
        {
            bool operator == (TestData1 const&) const = default;
            int a;
            float b;
            char c;
        };
        struct TestData2
        {
            TestData2() = delete;
            TestData2(float c)
                : c{c}
            {}
            ~TestData2()
            {
            }
            bool operator == (TestData2 const&) const = default;
            int a{};
            char b{};
            float c;
            double d{0.0};
        };
        qrius::RingElement<TestData1, TestData2, std::size_t> ring_element;
        static_assert(sizeof(ring_element) == qrius::cacheline_size*3);

        ring_element.construct<TestData1>(1, 0.0, '1'); // Must be wrapped in a guard for exception safety
        assert((TestData1{1, 0.0, '1'} == ring_element.get<TestData1>()));
        ring_element.destroy<TestData1>();

        ring_element.construct<TestData2>(0.0);
        assert((TestData2{0.0} == ring_element.get<TestData2>()));
        ring_element.destroy<TestData2>();
        { 
            std::array<qrius::RingElement<TestData1, int, float, char>, 2> buffer;
            buffer[0].construct<TestData1>(1, 0.0, '1');
            buffer[0].construct<int>(1);
            buffer[0].construct<float>(0.0);
            buffer[0].construct<char>('1');
            assert((buffer[0].get<TestData1>() == TestData1{1, 0.0, '1'}));
            assert((buffer[0].get<int>() == 1));
            buffer[1].construct<TestData1>(1, 0.1, '1');
            buffer[1].construct<int>(1);
            buffer[1].construct<float>(0.0);
            buffer[1].construct<char>('1');
        }
    }
}

static void ringbuffer_test() noexcept
{
    {
        qrius::RingBuffer<1024, std::size_t> ring_buffer;
        ring_buffer.construct_at(0, 1);
        ring_buffer.destroy_at(0);
        ring_buffer.construct_at(0, 113);
        assert(ring_buffer.get(0) == 113);
        ring_buffer.construct_at(1, 37);
        assert(ring_buffer.get(1) == 37);
    }
    {
        struct TestData
        {
            TestData(int a)
                : a{a}
            {}
            virtual ~TestData()
            {}

            bool operator == (TestData const&) const = default;

            int a;
            int b{0};
        };
        {
            qrius::RingBuffer<1024, TestData> ring_buffer;
            ring_buffer.construct_at(0, 1);
            ring_buffer.destroy_at(0);
            ring_buffer.construct_at(0, 37);
            assert(ring_buffer.get(0) == TestData{37});
            ring_buffer.destroy_at(0);
        }
        {
            qrius::RingBuffer<1024, std::size_t, TestData, double> ring_buffer;
            ring_buffer.construct_at<std::size_t>(0, 37);
            ring_buffer.construct_at<double>(0, 0.0);
            ring_buffer.construct_at<TestData>(0, 73);
            assert(ring_buffer.get<std::size_t>(0) == 37);
            assert((ring_buffer.get<TestData>(0) == TestData{73}));
            assert((ring_buffer.get<double>(0) == 0.0));
            ring_buffer.destroy_at<TestData>(0);
        }
    }
}

int main()
{
    uninit_storage_test();
    ringelement_test();
    ringbuffer_test();
}