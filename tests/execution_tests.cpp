// MinGW/GCC smoke test for the C++26-era concurrency stack OpenF4 relies on:
//  - libstdc++ parallel algorithms (std::execution::par) backed by oneTBB, and
//  - stdexec P2300 senders, when available.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <execution>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

#if defined(OPENF4_HAS_STDEXEC)
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#endif

#if defined(__GLIBCXX__) && !defined(_PSTL_PAR_BACKEND_TBB)
#error "libstdc++ parallel algorithms are not using the TBB backend (std::execution::par would be serial)"
#endif

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
}

int main()
{
    // Parallel algorithms produce correct results.
    std::vector<long long> values(1 << 20);
    std::iota(values.begin(), values.end(), 0LL);
    const long long sum = std::reduce(std::execution::par, values.begin(), values.end(), 0LL);
    const long long n = static_cast<long long>(values.size());
    check(sum == n * (n - 1) / 2, "par reduce sum");

    std::vector<long long> doubled(values.size());
    std::transform(std::execution::par_unseq, values.begin(), values.end(), doubled.begin(),
        [](long long v) { return v * 2; });
    check(doubled.back() == (n - 1) * 2, "par_unseq transform");

    // ...and actually fan out across threads on a multi-core machine.
    std::mutex idsMutex;
    std::set<std::thread::id> ids;
    std::vector<int> work(4096);
    std::for_each(std::execution::par, work.begin(), work.end(), [&](int&) {
        volatile double x = 0;
        for (int i = 0; i < 2000; ++i) x = x + i * 0.5;
        std::scoped_lock lock(idsMutex);
        ids.insert(std::this_thread::get_id());
    });
    std::printf("std::execution::par used %zu thread(s) (hw=%u)\n", ids.size(), std::thread::hardware_concurrency());
    if (std::thread::hardware_concurrency() > 1) check(ids.size() > 1, "par for_each ran on >1 thread");

#if defined(OPENF4_HAS_STDEXEC)
    // Senders: schedule work on a pool, chain continuations, join with when_all.
    exec::static_thread_pool pool(4);
    auto sched = pool.get_scheduler();
    auto left = stdexec::schedule(sched) | stdexec::then([] { return 20; });
    auto right = stdexec::schedule(sched) | stdexec::then([] { return 22; });
    auto joined = stdexec::when_all(std::move(left), std::move(right))
        | stdexec::then([](int a, int b) { return a + b; });
    auto [answer] = stdexec::sync_wait(std::move(joined)).value();
    check(answer == 42, "stdexec when_all/then/sync_wait");

    std::atomic<int> bulkCount{0};
    auto bulkWork = stdexec::schedule(sched)
        | stdexec::bulk(stdexec::par, 64, [&](int) { bulkCount.fetch_add(1, std::memory_order_relaxed); });
    stdexec::sync_wait(std::move(bulkWork));
    check(bulkCount.load() == 64, "stdexec bulk");
    std::printf("stdexec senders: ok\n");
#else
    std::printf("stdexec not available; senders not tested\n");
#endif

    if (failures == 0) std::printf("execution tests passed\n");
    return failures == 0 ? 0 : 1;
}