// This is for profiling and demonstratin
#define COUNT_STEALING
#define PAPSO2_TRACK_CONVERGENCY
#include "papso2_test.h"
#include <cstdio>

static constexpr auto schwefel_12 = test_functions::functions[1];
#pragma optimize("", off)
template <size_t Scale> requires (Scale > 0)
double scaled_schwefel_12(iter beg, iter end) {
	volatile double result = 0;
	for (size_t i = 0; i < Scale; ++i) {
		result = schwefel_12(beg, end);
	}
	return result;
}
#pragma optimize("", on)

int main(int argc, const char* argv[]) {
	if (argc <= 2) {
		std::printf("Usage: papso [number of subswarms] [iterations/task]\n");
		return -1;
	}

	size_t fork_count = std::stoul(std::string{ argv[1] });
	size_t iter_per_task = std::stoul(std::string{ argv[2] });

	size_t thread_count = argc < 4
		? fork_count
		: std::stoul(std::string{ argv[3] });

	hungbiu::hb_executor etor(thread_count);
	optimization_problem_t problem{ &scaled_schwefel_12<4>
									, test_functions::bounds[1]
									, test_functions::dimensions[1] };
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, 2, 48, 5000>;
	parallel_async_pso_benchmark<papso_t>(etor, fork_count, iter_per_task, problem, test_functions::function_names[1]);
	etor.done();
}


