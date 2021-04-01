// This is for profiling and demonstrating
#include "papso2_test.h"
#include <cstdio>

static constexpr auto schwefel_12 = test_functions::functions[1];
#pragma optimize("", off)
double scaled_schwefel_12(iter beg, iter end) {
	for (int i = 0; i < 9; ++i) {
		schwefel_12(beg, end);
	}
	return schwefel_12(beg, end);
}

#pragma optimize("", on)

int main(int argc, const char* argv[]) {
	if (argc <= 2) {
		printf("Usage: papso [number of subswarms] [iterations/task]\n");
		return -1;
	}

	size_t fork_count = std::stoul(std::string{ argv[1] });
	size_t iter_per_task = std::stoul(std::string{ argv[2] });

	hungbiu::hb_executor etor(fork_count);
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, 40>;
	papso_t::optimization_problem_t problem{ test_functions::functions[3]
										, test_functions::bounds[3]
										, test_functions::dimensions[3] };

	parallel_async_pso_benchmark<papso_t>(etor, fork_count, iter_per_task, problem, test_functions::function_names[3]);
	etor.done();
	printf("steal count: %llu\n", etor.get_steal_count());
}


