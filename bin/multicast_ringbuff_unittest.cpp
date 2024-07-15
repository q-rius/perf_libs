// TODO: Convert to GTest.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "multicast_ringbuff.hpp"
#include <cstdint>
#include <iostream>
#include <thread>
#include <random>
#include <barrier>

static constexpr void perf_utils_test() noexcept
{
    {
        static_assert(alignof(int) == 4);
        static_assert(alignof(std::uint64_t) == 8);
        static_assert(qrius::cacheline_align<int> == qrius::cacheline_size);
        struct AlignmentTest
        {
            alignas(256) int test;
        };
        static_assert(qrius::cacheline_align<AlignmentTest> == 256UL);
    }
    {
        static_assert(qrius::cacheline_padding<int> == qrius::cacheline_size - sizeof(int));
        static_assert(qrius::cacheline_padding<char[130]> == 62UL);
        static_assert(!qrius::cacheline_padding<char[192]>);
        static_assert(qrius::cacheline_padding<char&> == qrius::cacheline_size - sizeof(char*));
        struct Data
        {
            char&   a;
            int&    b;
            double  c;
            int     d;
            char    e;
        };
        static_assert(qrius::cacheline_padding<char&, int&, double, int, char> == qrius::cacheline_size - sizeof(Data));
        static_assert(sizeof(qrius::CachelinePadd<char&, int&, double, int, char>) == qrius::cacheline_size - sizeof(Data));
        {
            struct PaddedData : Data
            {
                qrius::CachelinePadd<char&, int&, double, int, char> padd{};
            };
            static_assert(sizeof(PaddedData) == qrius::cacheline_size);
        }
        {
            struct PaddedData : Data
            {
                qrius::CachelinePadd<Data> padd{};
            };
            static_assert(sizeof(PaddedData) == qrius::cacheline_size);
        }
    }
}

static void blocking_ringbuff_test()
{
    {
        using MCRingBuff = qrius::RingBuff<std::uint64_t, 1, 16, false>;
        static_assert(sizeof(MCRingBuff::Writer) == 128UL);
        static_assert(sizeof(MCRingBuff::Reader) == 128UL);
        static_assert(sizeof(MCRingBuff) == 384UL);
    }
    {
        using MCRingBuff = qrius::RingBuff<std::uint64_t, 4, 1, false>;
        static_assert(MCRingBuff::capacity == 1);
        static_assert(MCRingBuff::reader_count == 4);
        MCRingBuff rb;
        auto& writer = rb.get_writer();
        assert(writer.write(0UL));
        assert(!writer.write(1UL));
    }
    {
        using MCRingBuff = qrius::RingBuff<std::uint64_t, 1, 2, false>;
        static_assert(MCRingBuff::capacity == 2);
        static_assert(MCRingBuff::reader_count == 1);
        MCRingBuff rb;
        auto& reader = rb.get_reader(0UL);
        assert(!reader.acquire());
        auto& writer = rb.get_writer();
        assert(writer.write(11UL));
        assert(writer.write(13UL));
        assert(!writer.write(14UL));
        assert(reader.acquire() == 1UL);
        assert(reader.read() == 11UL);
        reader.pop();
        reader.commit();
        assert(writer.write(13UL));
    }
    {
        using MCRingBuff = qrius::RingBuff<std::uint64_t, 4, 1, false>;
        static_assert(MCRingBuff::capacity == 1);
        static_assert(MCRingBuff::reader_count == 4);
        MCRingBuff rb;
        {
            auto& writer = rb.get_writer();
            auto slots = writer.acquire();
            assert(slots == 1);
            MCRingBuff::ValueType value = 0;
            for(auto slot=0UL; slot != slots; ++slot)
            {
                writer.emplace(value++);
            }
            writer.commit();
            assert(writer.acquire() == 0); // Can't write anymore untill all readers have progressed
        }
        {
            for(auto i=0UL; i != MCRingBuff::reader_count; ++i)
            {
                auto& reader = rb.get_reader(i);
                auto slots = reader.acquire();
                assert(slots == 1);
                MCRingBuff::ValueType value = 0;
                for(auto slot=0UL; slot != slots; ++slot)
                {
                    auto read_value = reader.read();
                    assert(read_value == value++);
                    reader.pop();
                }
                reader.commit();
                assert(reader.acquire() == 0);
            }
            assert(rb.get_writer().acquire() == 1);
        }
    }
    {
        // char array buffer not supported directly.
        // https://cplusplus.github.io/LWG/issue3436
        // A possible but poor alternative is std::array
        using MCRingBuff = qrius::RingBuff<std::array<char, 5>, 4, 1, false>;
        static_assert(MCRingBuff::capacity == 1);
        static_assert(MCRingBuff::reader_count == 4);
        MCRingBuff rb;
        {
            auto& writer = rb.get_writer();
            auto slots = writer.acquire();
            assert(slots == 1);
            MCRingBuff::ValueType value = {'H','e','l','l','0'};
            for(auto slot=0UL; slot != slots; ++slot)
            {
                writer.emplace(value);
            }
            writer.commit();
            assert(writer.acquire() == 0);
        }
    }
    {
        // char array buffer not supported directly.
        // https://cplusplus.github.io/LWG/issue3436
        // A possible but poor alternative is std::array
        using MCRingBuff = qrius::RingBuff<int*, 4, 1, false>;
        static_assert(MCRingBuff::capacity == 1);
        static_assert(MCRingBuff::reader_count == 4);
        MCRingBuff rb;
        int int_value = 1001;
        {
            auto& writer = rb.get_writer();
            auto slots = writer.acquire();
            assert(slots == 1);
            MCRingBuff::ValueType value = &int_value;
            for(auto slot=0UL; slot != slots; ++slot)
            {
                writer.emplace(value);
            }
            writer.commit();
            assert(writer.acquire() == 0);
        }
        {
            for(auto i=0UL; i!=MCRingBuff::reader_count; ++i)
            {
                auto& reader = rb.get_reader(i);
                auto slots = reader.acquire();
                assert(slots == 1);
                for(auto slot=0UL; slot != slots; ++slot)
                {
                    auto value = reader.read();
                    assert(std::addressof(int_value) == value);
                    reader.pop();
                }
                reader.commit();
            }
        }
    }
}

template<std::size_t reader_count=4>
static void concurrent_randomized_test(std::size_t test_items=1'000'000UL) noexcept
{
    struct Data
    {
        std::uint64_t a{0};
        std::uint64_t b{0};
        std::uint64_t c{0};
        bool operator == (Data const& rhs) const = default;
    };
    using MCRingBuff = qrius::RingBuff<Data, reader_count, 32, false>;
    static_assert(MCRingBuff::capacity == 32);
    static_assert(MCRingBuff::reader_count == reader_count);
    MCRingBuff rb;
    auto rand_seed = std::random_device{}();
    std::jthread producer(
        [rand_seed, &rb, test_items]()
        {
            auto &writer = rb.get_writer();
            std::mt19937 mt{rand_seed};
            auto item = 0UL;
            while(item < test_items)
            {
                auto slots = 0UL;
                while(!(slots = writer. template acquire<4UL>())) asm("pause");
                for(auto slot = 0UL; slot != slots; ++slot)
                {
                    auto rand_val = mt();
                    writer.emplace(Data{rand_val, rand_val, rand_val});
                }
                writer.commit();
                item += slots;
            }
        }
    );
    std::array<std::jthread, MCRingBuff::reader_count> reader_threads;
    auto reader_index = 0UL;
    for(auto &reader_thread : reader_threads)
    {
        reader_thread = std::jthread(
            [rand_seed, &rb, reader_index, test_items]()
            {
                auto& reader = rb.get_reader(reader_index);
                std::mt19937 mt{rand_seed};
                auto item = 0UL;
                while(item < test_items)
                {
                    auto slots = 0UL;
                    while(!(slots = reader. template acquire<2UL>())) asm("pause");
                    for(auto slot = 0UL; slot != slots; ++slot)
                    {
                        auto rand_val = mt();
                        assert((Data{rand_val, rand_val, rand_val} == reader.read()));
                        reader.pop();
                    }
                    reader.commit();
                    item += slots;
                }
            }
        );
        ++reader_index;
    }
}

static void seqlock_test()
{
    {
        qrius::Seqlock<int*> seqlock;
        seqlock.emplace(0UL, nullptr);
        assert(seqlock.read_ready(0UL));
        auto [result, seqno] = seqlock.read();
        assert(!result);
        assert(!seqno);
    }
    {
        struct Data
        {
            std::uint64_t a{0};
            std::uint64_t b{0};
            std::uint64_t c{0};
            bool operator == (Data const& rhs) const = default;
        };
        qrius::Seqlock<Data> seqlock;
        seqlock.emplace(100UL, Data{1, 1, 1});
        assert(seqlock.read_ready(35UL));
        auto [result, seqno] = seqlock.read();
        assert((result == Data{1, 1, 1}));
    }
}

template<std::size_t reader_count=4UL>
void concurrent_seqlock_test(std::size_t test_items)
{
    struct Data
    {
        std::uint64_t a{0};
        std::uint64_t b{0};
        std::uint64_t c{0};
        bool operator == (Data const& rhs) const = default;
    };
    qrius::Seqlock<Data> seqlock;
    static_assert(sizeof(seqlock) < qrius::cacheline_size);
    std::barrier start_barrier(reader_count + 1);
    std::jthread writer(
        [&start_barrier, &seqlock, test_items]()
        {
            start_barrier.arrive_and_wait();
            for(auto item=0UL; item != test_items; ++item)
            {
                seqlock.emplace(item, Data{item, item, item});
            }
        });

    std::array<std::jthread, reader_count> readers;
    for(auto& reader_thread : readers)
    {
        reader_thread = std::jthread(
            [&start_barrier, &seqlock, test_items]()
            {
                auto item = 0UL;
                start_barrier.arrive_and_wait();
                while(item < test_items)
                {
                    while(!seqlock.read_ready(item));
                    Data result;
                    std::tie(result, item) = seqlock.read();
                    assert((result == Data{item, item, item}));
                    ++item;
                }
            });
    }
}

template<std::size_t reader_count=4UL>
void concurrent_non_blocking_ringbuff_test(std::size_t test_items=1'000'000UL)
{
    struct Data
    {
        std::uint64_t a{0};
        std::uint64_t b{0};
        std::uint64_t c{0};
        bool operator == (Data const& rhs) const = default;
    };
    using MCRingBuff = qrius::RingBuff<Data, reader_count, 32, true>;
    static_assert(MCRingBuff::capacity == 32);
    static_assert(MCRingBuff::reader_count == reader_count);
    static_assert(alignof(MCRingBuff) >= qrius::cacheline_size);
    static_assert(!(sizeof(MCRingBuff)%qrius::cacheline_size));
    MCRingBuff rb;

    std::barrier start_barrier(reader_count + 1);
    std::jthread writer(
        [&start_barrier, &rb, test_items]()
        {
            auto& writer = rb.get_writer();
            start_barrier.arrive_and_wait();
            for(auto item=0UL; item != test_items; ++item)
            {
                writer.emplace(Data{item, item, item});
            }
        });

    std::array<std::jthread, reader_count> readers;
    auto reader_index = 0UL;
    for(auto& reader_thread : readers)
    {
        reader_thread = std::jthread(
            [&start_barrier, reader_index, &rb, test_items]()
            {
                auto& reader = rb.get_reader(reader_index);
                auto item = 0UL;
                start_barrier.arrive_and_wait();
                while(item < test_items)
                {
                    while(!reader.data_available()) asm("pause");
                    Data result = reader.read_data();
                    assert(item <= result.a);
                    assert(result.a == result.b && result.b == result.c);
                    item = result.a + 1;
                }
            });
        ++reader_index;
    }
}

static void non_blocking_ringbuff_test()
{
    {
        struct TrivialData
        {
            int a;
            int b;
        };
        static_assert(std::is_trivially_constructible_v<TrivialData, int, int>);
        using MCRingBuff = qrius::RingBuff<TrivialData, 1, 1, true>;
        MCRingBuff ring_buff;
        ring_buff.get_writer().emplace(0, 1);
    }
    {
        struct NonTrivialData
        {
            NonTrivialData() noexcept
                : a{1}
            {}
            int a;
        };
        using MCRingBuff = qrius::RingBuff<NonTrivialData, 1, 1, true>;
        MCRingBuff ring_buff;
#if 0
        ring_buff.get_writer().emplace(); // Not accepted since default constructor is not trvial.
#endif
        ring_buff.get_writer().emplace(NonTrivialData{}); // Accepted since copy constructor is trivial
    }
    {
        using MCRingBuff = qrius::RingBuff<std::uint64_t, 1, 1, true>;
        MCRingBuff ring_buff;
        auto &writer = ring_buff.get_writer();
        static_assert(sizeof(writer) == 64UL);
        static_assert(sizeof(std::array<MCRingBuff::Writer, 2>) == 64*2);
        auto &reader = ring_buff.get_reader(0UL);
        static_assert(sizeof(reader) == 64UL);
        static_assert(sizeof(ring_buff) == 64 + 64 + 64);
        writer.emplace(13UL);
        assert(reader.data_available());
        auto [data, exp_seqno, actual_seqno] = reader.read();
        assert(data == 13UL);
        assert(actual_seqno == 0UL);
        assert(exp_seqno == actual_seqno);
        assert(!reader.data_available());
    }
    {
        struct Data
        {
            std::uint64_t a{0};
            std::uint64_t b{0};
            std::uint64_t c{0};
            bool operator == (Data const& rhs) const = default;
        };
        using MCRingBuff = qrius::RingBuff<Data, 2, 1, true>;
        MCRingBuff ring_buff;
        auto &writer = ring_buff.get_writer();
        writer.emplace(Data{13, 13, 13});
        {
            auto &reader = ring_buff.get_reader(0UL);
            assert(reader.data_available());
            assert((reader.read_data() == Data{13, 13, 13}));
            assert(!reader.data_available());
        }
        {
            auto &reader = ring_buff.get_reader(1UL);
            assert(reader.data_available());
            assert((reader.read_data() == Data{13, 13, 13}));
            assert(!reader.data_available());
        }
    }
}

///
/// throws if the host doesn't have adequate number of huge pages enabled
/// Create a few huge pages before the test is run. For ref: huge_page.sh.
///
void huge_page_alloc_test()
{
    {
        auto& seq_lock = *qrius::make_unique_on_huge_pages<qrius::Seqlock<int>>();
        seq_lock.emplace(0, 42);
        assert((seq_lock.read() == std::pair{42, 0UL}));
    }
    {
        struct Data
        {
            std::uint64_t a{0};
            std::uint64_t b{0};
            std::uint64_t c{0};
            bool operator == (Data const& rhs) const = default;
        };
        using MCRingBuff = qrius::RingBuff<Data, 1, 1024, false>;
        MCRingBuff& rb = *qrius::make_unique_on_huge_pages<MCRingBuff>();
        {
            auto& writer = rb.get_writer();
            auto slots = writer.acquire<MCRingBuff::capacity>();
            assert(slots == MCRingBuff::capacity);
            for(auto slot=0UL; slot != slots; ++slot)
            {
                writer.emplace(slot, slot, slot);
            }
            writer.commit();
            assert(writer.acquire() == 0); // Can't write anymore untill all readers have progressed
        }
        {
            auto& reader = rb.get_reader(0);
            auto slots = reader.acquire<MCRingBuff::capacity>();
            assert(slots == MCRingBuff::capacity);
            for(auto slot=0UL; slot != slots; ++slot)
            {
                auto [a, b, c]= reader.read();
                assert(a == slot);
                assert(b == slot);
                assert(c == slot);
                reader.pop();
            }
            reader.commit();
            assert(reader.acquire() == 0);
        }
    }
}

int main()
{
    perf_utils_test();

    blocking_ringbuff_test();
    concurrent_randomized_test(); // 4 readers, 1M items on a ring buffer with 32 slots
    concurrent_randomized_test<100>(1'000UL); // 100 readers, 1000 items on a ring buffer with 32 slots
    concurrent_randomized_test<10>(10'000UL); // 10 readers, 10000 items on a ring buffer with 32 slots

    seqlock_test();
    concurrent_seqlock_test(1'000'000UL);
    concurrent_seqlock_test<100>(1'000UL);
    concurrent_seqlock_test<10>(10'000UL);

    non_blocking_ringbuff_test();
    concurrent_non_blocking_ringbuff_test();
    concurrent_non_blocking_ringbuff_test<100>(1000UL);
    concurrent_non_blocking_ringbuff_test<10>(); // 10 readers, 10000 items on a ring buffer with 32 slots

    huge_page_alloc_test();
    return 0;
}