#ifndef _PAPSO_TEST
#define _PAPSO_TEST
#include "papso2.h"
#include "test_functions.h"

template <typename papso_t>
void parallel_async_pso_benchmark(
	hungbiu::hb_executor& etor
	, std::size_t fork_count
	, std::size_t iter_per_task
	, optimization_problem_t problem
	, const char* const msg) {

	for (int i = 0; i < 10; ++i) {
		auto result = papso_t::parallel_async_pso(etor, fork_count, iter_per_task, problem);
		auto [v, pos] = result.get(); // Could be wasting?
		printf_s("\npar async pso @%s: %lf\n", msg, v);
#ifdef COUNT_STEALING
		std::printf("steal count: %llu\n", etor.get_steal_count());
#endif
		printf("\n");
	}
}

#endif