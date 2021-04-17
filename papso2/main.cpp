// This is for profiling and demonstratin
#define COUNT_STEALING
//#define PAPSO2_TRACK_CONVERGENCY
#include "papso2_test.h"
#include <cstdio>

template <size_t Scale> requires (Scale > 0)
struct scaled_rosenbrock {
	static double function(iter beg, iter end) {
		static constexpr auto rosenbrock = test_functions::functions[2];
		volatile double result = 0;
		for (int i = 0; i < Scale; ++i) {
			//benchmark::DoNotOptimize(  );
			result = rosenbrock(beg, end);
		}
		return result;
	}

	static constexpr optimization_problem_t problem{
		&function
		, test_functions::bounds[2]
		, test_functions::dimensions[2]
	};
};

int main(int argc, const char* argv[]) {
	if (argc <= 2) {
		std::printf("Usage: papso [number of subswarms] [iterations/task] [thread_count(optional)]\n");
		return -1;
	}

	size_t fork_count = std::stoul(std::string{ argv[1] });
	size_t iter_per_task = std::stoul(std::string{ argv[2] });

	size_t thread_count = argc < 4
		? fork_count
		: std::stoul(std::string{ argv[3] });

	hungbiu::hb_executor etor(thread_count);
	optimization_problem_t problem = scaled_rosenbrock<50>::problem;
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, 2, 100, 5000>;
	parallel_async_pso_benchmark<papso_t>(etor, fork_count, iter_per_task, problem, test_functions::function_names[1]);
	etor.done();
}


