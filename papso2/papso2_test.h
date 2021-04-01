#ifndef _PAPSO_TEST
#define _PAPSO_TEST
#include "papso2.h"
#include "test_functions.h"

template <typename papso_t>
void parallel_async_pso_benchmark(
	hungbiu::hb_executor& etor
	, std::size_t fork_count
	, std::size_t iter_per_task
	, typename papso_t::optimization_problem_t problem
	, const char* const msg) {
	auto result = papso_t::parallel_async_pso(etor, fork_count, iter_per_task, problem);
	auto [v, pos] = result.get(); // Could be wasting?
	printf_s("\npar async pso @%s: %lf\n", msg, v);
	printf("\n");
}

#endif