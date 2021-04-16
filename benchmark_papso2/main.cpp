#pragma comment ( lib, "Shlwapi.lib" )
#include "../../google_benchmark/include/benchmark/benchmark.h"
#include "../papso2/executor.h"
#include "../papso2/papso2_test.h"

static constexpr auto schwefel_12 = test_functions::functions[1];
template <size_t Scale> requires (Scale > 0)
double scaled_schwefel_12(iter beg, iter end) {
	double result = 0;
	for (int i = 0; i < Scale; ++i) {
		benchmark::DoNotOptimize( result = schwefel_12(beg, end) );
	}
	return result;
}

static void benchmark_executor_create(benchmark::State& state) {
	const auto count = state.range(0);
	for (auto _ : state) {
		benchmark::DoNotOptimize(hungbiu::hb_executor{ static_cast<size_t>(count) });
	}
}
//BENCHMARK(benchmark_executor_create)
//->Unit(benchmark::kMicrosecond)
//->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8);


// Bench speed of optimizing test functions suite
// Args: [fork_count] [iter_per_task] [thread_count]
static void benchmark_papso(benchmark::State& state) {
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, 2, 48, 5000>;

	size_t fork_count = state.range(0);
	size_t iter_per_task = state.range(1);
	hungbiu::hb_executor etor{ static_cast<size_t>(state.range(2)) };

	const optimization_problem_t problem{
			&scaled_schwefel_12<4>
			, test_functions::bounds[1]
			, test_functions::dimensions[1]
	};

    for (auto _ : state) {
		auto result = papso_t::parallel_async_pso(etor, fork_count, iter_per_task, problem);
		benchmark::DoNotOptimize(result.get());
    }        
}

BENCHMARK(benchmark_papso)
->Iterations(1)
->Repetitions(5)
->Unit(benchmark::kMillisecond)
->Args({ 4, 1, 2 })
->Args({ 4, 1, 3 })
->Args({ 4, 1, 4 });
//->Args({ 5, 500 })
//->Args({ 6, 500 })
//->Args({ 7, 500 })
//->Args({ 8, 500 });

// Args: [function idx] [dimensions] [iterations]
static void benchmark_test_functions(benchmark::State& state) {
	const auto idx = state.range(0);
	const auto dim = state.range(1);
	const auto itr = state.range(2);
	const auto min = test_functions::bounds[idx].first;
	const auto max = test_functions::bounds[idx].second;
	const auto diff = max - min;
	canonical_rng rng;

	std::vector<double> vec(dim);
	std::generate(vec.begin(), vec.end(), [&]() {
		return min + rng() * diff;	});

	for (auto _ : state) {
		for (int i = 0; i < itr; ++i) {
			test_functions::functions[idx](vec.cbegin(), vec.cend());
		}
	}
}
//
//BENCHMARK(benchmark_test_functions)
//->Args({ 1, 30, 1 })
//->Args({ 1, 30, 100 })
//->Args({ 1, 100, 1 })
//->Args({ 1, 100, 100 });


template <int N> // N iteartions, Args: [fork_count] [itr_per_task] [dimensions] 
void benchmark_stealing_efficiency(benchmark::State& state) {	
	// A costly object function
	static constexpr auto schwefel_12 = test_functions::functions[1];
	func_t scaled_schwefel_12 = [](iter beg, iter end) {
		for (int i = 0; i < N - 1; ++i) {
			benchmark::DoNotOptimize(schwefel_12(beg, end));
		}
		return schwefel_12(beg, end);
	};

	// Prep args
	const auto dimensions = state.range(2);
	const papso::optimization_problem_t problem{
				scaled_schwefel_12,
				test_functions::bounds[1],
				dimensions
	};
	const auto fork_count = state.range(0);
	const auto itr_per_task = state.range(1);

	// Bench
	hungbiu::hb_executor etor(fork_count);
	for (auto _ : state) {
		auto result = papso::parallel_async_pso(etor, fork_count, itr_per_task, problem);
		benchmark::DoNotOptimize(result.get());
	}
}

//BENCHMARK_TEMPLATE(benchmark_stealing_efficiency, 10)
//->Unit(benchmark::kMillisecond)
//->Args({ 1, 100, 30 })
//->Args({ 2, 100, 30 })
//->Args({ 3, 100, 30 })
//->Args({ 4, 100, 30 })
//->Args({ 5, 100, 30 })
//->Args({ 6, 100, 30 })
//->Args({ 7, 100, 30 })
//->Args({ 8, 100, 30 });

// Args: [thread_count] [fork_count] [itr_per_task]
static void benchmark_stealing(benchmark::State& state) {	
	const auto thread_count = state.range(0);
	const auto fork_count = state.range(1);
	const auto itr_per_task = state.range(2);
	const auto func_index = 1;
	// Bench
	optimization_problem_t problem{
		test_functions::functions[func_index],
		test_functions::bounds[func_index],
		test_functions::dimensions[func_index]
	};
	hungbiu::hb_executor etor(thread_count);
	for (auto _ : state) {
		auto result = papso::parallel_async_pso(etor, 8, itr_per_task, problem);
		benchmark::DoNotOptimize(result.get());
	}
}

//BENCHMARK(benchmark_stealing)
//->Unit(benchmark::kMillisecond)
//->Args({ 4, 4, 500 })
//->Args({ 5, 5, 500 })
//->Repetitions(10);

// Args: [fork_count]
template <int sz> requires (sz > 0)
double time_scaled_func(iter beg, iter end) { // 420ns
	for (int i = 0; i < sz - 1; ++i) {
		test_functions::functions[3](beg, end);
	}
	return test_functions::functions[3](beg, end);
}
template <typename buffer_t, size_t nb_sz>
void benchmark_particle_communication(benchmark::State& state) {
	using papso_t = basic_papso<buffer_t, nb_sz, nb_sz>; // global topology

	const auto fork_count = state.range(0);
	
	optimization_problem_t problem{
		test_functions::functions[3],
		test_functions::bounds[3],
		test_functions::dimensions[3]
	};

	hungbiu::hb_executor etor(fork_count);

	for (auto _ : state) {
		auto result = papso::parallel_async_pso(etor, fork_count, 5000, problem);
		benchmark::DoNotOptimize(result.get());
	}
}
using naive_buffer = hungbiu::naive_spmc_buffer<vec_t>;
using my_buffer = hungbiu::spmc_buffer<vec_t>;/*

// Neighbor hood size
BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 2u, 1)
->Unit(benchmark::kMillisecond)->Arg(4)->Iterations(10)->Repetitions(10);
BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 12u, 1)
->Unit(benchmark::kMillisecond)->Arg(4)->Iterations(10)->Repetitions(10);
BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 24u, 1)
->Unit(benchmark::kMillisecond)->Arg(4)->Iterations(10)->Repetitions(10);
BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 36u, 1)
->Unit(benchmark::kMillisecond)->Arg(4)->Iterations(10)->Repetitions(10);*/

// Communication cost - Forks
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 40u, 1)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Repetitions(10)
//->Arg(2)->Arg(4)->Arg(6)->Arg(8);
//
//
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 40u, 1)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Repetitions(10)
//->Arg(2)->Arg(4)->Arg(6)->Arg(8);

// Commnucation cost - Swarm size
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 40u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 50u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 60u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 70u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 80u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 40u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 50u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 60u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 70u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 80u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);




BENCHMARK_MAIN();