#ifndef _EXECUTOR
#define _EXECUTOR
#include <type_traits>
#include <memory>
#include <vector>
#include <cstddef>
#include <random>
#include <cassert>
#include <concepts>
#include <future>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "concurrent_std_deque.h"
#include "../../concurrent_data_structures/concurrent_data_structures/array_blocking_queue.h"
//#define PRINT_ETOR
#ifdef PRINT_ETOR
#include <cstdio>
#endif
#define COUNT_STEALING

// lazy spin up + cv
namespace hungbiu
{		
	class hb_executor
	{			
	public:	// Template aliases used by hb_executor
		template <typename T>
		using promise_t = std::promise<T>;
		template <typename T>
		using future_t = std::future<T>;
		class worker_handle; // Forward declaration

	private:
		// --------------------------------------------------------------------------------
		// task_t
		// Erase function object's type but leaves return type and arg type(s)
		// --------------------------------------------------------------------------------
		template <typename R>
		using task_t = std::packaged_task<R(worker_handle&)>;
		template <typename F, typename R>
		static task_t<R> make_task(F&& func)
		{
			using F_decay = std::decay_t<F>;
			return task_t<R>(std::forward<F_decay>(func));
		}

		// --------------------------------------------------------------------------------
		// task_wrapper
		// Erase all type info
		// --------------------------------------------------------------------------------    

		// Provide task interface
		struct task_wrapper_alloc_concept
		{
			virtual ~task_wrapper_alloc_concept() = default; // Require a dtor definition for
													   // the derived class to instantiate
			virtual void run(worker_handle&) = 0;
		};

		// Provide storage for an actual task
		template <typename T>
		struct task_wrapper_alloc_model : public task_wrapper_alloc_concept
		{
			T task_;
		public:
			~task_wrapper_alloc_model() {}
			task_wrapper_alloc_model(T&& f) :
				task_(std::forward<T>(f)) {}
			task_wrapper_alloc_model(task_wrapper_alloc_model&& oth) :
				task_(std::move(oth.task_)) {}
			task_wrapper_alloc_model& operator=(task_wrapper_alloc_model&& rhs)
			{
				if (this != &rhs) {
					task_ = std::move(rhs.task_);
				}
				return *this;
			}

			task_wrapper_alloc_model(const task_wrapper_alloc_model&) = delete;
			task_wrapper_alloc_model& operator=(const task_wrapper_alloc_model&) = delete;

			void run(worker_handle& h) override
			{
				task_.operator()(h);
			}
		};

		// Provide total type erasure to support heterogeneous tasks
		// Always heap-allocated
		class task_wrapper_alloc
		{
			std::unique_ptr<task_wrapper_alloc_concept> task_vtable_;
		public:
			task_wrapper_alloc() : task_vtable_(nullptr) {}
			template <typename T>
			task_wrapper_alloc(T&& t)
			{
				using model_t = task_wrapper_alloc_model<T>;
				task_vtable_ = std::make_unique<model_t>(std::move(t));
			}
			task_wrapper_alloc(task_wrapper_alloc&& oth) noexcept :
				task_vtable_(std::move(oth.task_vtable_)) {}
			~task_wrapper_alloc() {}
			task_wrapper_alloc& operator=(task_wrapper_alloc&& rhs) noexcept
			{
				if (this != &rhs) {
					task_vtable_ = std::move(rhs.task_vtable_);
				}
				return *this;
			}

			bool valid() const noexcept
			{
				return task_vtable_.get();
			}
			explicit operator bool() const noexcept
			{
				return task_vtable_.get();
			}
			void run(worker_handle& h)
			{
				assert(valid());
				task_vtable_->run(h);
			}
		};

		// --------------------------------------------------------------------------------
		// task_wrapper with small object opitimization
		// --------------------------------------------------------------------------------
		struct task_wrapper_concept_sso
		{
			void (*_destructor)(void*) noexcept;
			void (*_move)(void*, void*) noexcept;
			void (*_run)(void*, worker_handle&);
		};

		static constexpr auto small_size = sizeof(void*) * 7;
		template <typename T>
		static constexpr bool is_small_object() { return sizeof(T) <= small_size; }

		template <typename T, bool Small = is_small_object<T>()>
		struct task_wrapper_model_sso;
		template <typename F>
		struct task_wrapper_model_sso<F, true>
		{
			F task_;

			template <typename U>
			task_wrapper_model_sso(U&& func) :
				task_(std::forward<F>(func)) {}
			task_wrapper_model_sso(task_wrapper_model_sso&& oth) :
				task_(std::move(oth.task_)) {}
			~task_wrapper_model_sso() = default;

			static void _destructor(void* p) noexcept
			{
				static_cast<task_wrapper_model_sso*>(p)->~task_wrapper_model_sso();
			}
			static void _move(void* lhs, void* rhs) noexcept
			{
				auto& model = *static_cast<task_wrapper_model_sso*>(rhs);
				new (lhs) task_wrapper_model_sso(std::move(model));
			}
			static void _run(void* p, worker_handle& h)
			{
				auto& t = static_cast<task_wrapper_model_sso*>(p)->task_;
				std::invoke(t, h);
			}
			static constexpr task_wrapper_concept_sso vtable_ = { _destructor, _move, _run };
		};
		template <typename F>
		struct task_wrapper_model_sso<F, false>
		{
			std::unique_ptr<F> ptask_;

			template <typename U>
			task_wrapper_model_sso(U&& func) :
				ptask_(std::make_unique<F>(std::forward<F>(func))) {}
			task_wrapper_model_sso(task_wrapper_model_sso&& oth) :
				ptask_(std::move(oth.ptask_)) {}
			~task_wrapper_model_sso() = default;

			static void _destructor(void* p) noexcept
			{
				static_cast<task_wrapper_model_sso*>(p)->~task_wrapper_model_sso();
			}
			static void _move(void* lhs, void* rhs) noexcept
			{
				auto& model = *static_cast<task_wrapper_model_sso*>(rhs);
				new (lhs) task_wrapper_model_sso(std::move(model));
			}
			static void _run(void* p, worker_handle& h)
			{
				auto pt = static_cast<task_wrapper_model_sso*>(p)->ptask_.get();
				if (pt) std::invoke(*pt, h);
			}
			static constexpr task_wrapper_concept_sso vtable_ = { _destructor, _move, _run };
		};

		class task_wrapper_sso
		{
			// Constness: the vtable is a const object!
			const task_wrapper_concept_sso* task_vtable_{ nullptr };
			std::aligned_storage_t<small_size> task_;
		public:
			task_wrapper_sso() : task_() {}
			template <typename T>
			task_wrapper_sso(T&& t)
			{
				using DecayT = std::decay_t<T>;
				using model_t = task_wrapper_model_sso<DecayT>;
				task_vtable_ = &model_t::vtable_;
				new (&task_) model_t(std::forward<DecayT>(t));
			}
			task_wrapper_sso(task_wrapper_sso&& oth) noexcept :
				task_vtable_(oth.task_vtable_)
			{
				task_vtable_->_move(&task_, &oth.task_);
			}
			~task_wrapper_sso()
			{
				if (task_vtable_) task_vtable_->_destructor(&task_);
			}
			task_wrapper_sso& operator=(task_wrapper_sso&& rhs) noexcept
			{
				if (this != &rhs) {
					// Destroy current task if there is one
					if (task_vtable_) task_vtable_->_destructor(&task_);

					// Copy rhs's vtable BEFORE moving the rhs's task
					// because rhs's task has an arbitrary type
					task_vtable_ = std::exchange(rhs.task_vtable_, nullptr);
					task_vtable_->_move(&task_, &rhs.task_);
				}
				return *this;
			}

			bool valid() const noexcept
			{
				return task_vtable_;
			}
			explicit operator bool() const noexcept
			{
				return valid();
			}
			void run(worker_handle& h)
			{
				if (task_vtable_) task_vtable_->_run(&task_, h);
			}
		};
				
		// Choose a flavor
		using task_wrapper = task_wrapper_sso;

		class worker; // Forward declaration because worker::get_handle() return a woker_handle object;
					  // Have to be PRIVATE
	public:		
		// --------------------------------------------------------------------------------
		// worker_handler
		// used by task object to submit work
		// --------------------------------------------------------------------------------		
		class worker_handle
		{
			worker* ptr_worker_;
		public:
			worker_handle(worker* worker) noexcept :
				ptr_worker_(worker) {}
			worker_handle(const worker_handle& oth) noexcept :
				ptr_worker_(oth.ptr_worker_) {}
			worker_handle& operator=(const worker_handle& rhs) noexcept
			{
				if (this != &rhs) ptr_worker_ = rhs.ptr_worker_;
				return *this;
			}

			template <typename T>
			static bool future_ready(const std::future<T>& fut)
			{
				assert(fut.valid());
				const auto status = fut.wait_for(std::chrono::nanoseconds{ 0 });
				return std::future_status::ready == status;
			}

			// Suspend current task to execute others
			template <typename R>
			R get(future_t<R>& fut)
			{
				while (!future_ready(fut)) {
					task_wrapper tw{};
					if (ptr_worker_->_pop(tw) ||
						ptr_worker_->_steal(tw)) {
						tw.run(*this);
					}
				}
				return fut.get();
			}

			// Submit a task to current thread, return future to obtain result
			template <
				typename F
				, typename R = std::invoke_result_t<F, worker_handle&> >
			requires std::invocable<F, hb_executor::worker_handle&>
				[[nodiscard]] future_t<R> execute_return(F&& func) const
			{
				auto t = make_task<F, R>(std::forward<F>(func));
				auto fut = t.get_future();
				ptr_worker_->_push(std::move(t));
				return fut;
			}

			// Submit a task to current thread and doesn't return result
			template <typename F>
			requires std::invocable<F, hb_executor::worker_handle&>
			void execute(F&& func) const
			{
				ptr_worker_->_push(std::move(func));
			}
		}; // end of class worker handle
	
	private:
		using rng_t = std::default_random_engine;
		// --------------------------------------------------------------------------------
		// A worker object is the working context of an OS thread
		// --------------------------------------------------------------------------------
		class alignas(64) worker
		{
			friend class worker_handle;
			template <typename T>
			using deque_t = concurrent_std_deque<T>;//concurrent_std_deque<T>;

			hb_executor* etor_;
			std::size_t index_;
			deque_t<task_wrapper> run_stack_;
			std::condition_variable_any cv_;
			std::mutex mtx_; // use this mutex to wait for condition

			// Waiting, Ready, Running
			// `Waiting` -> `Ready`:   Dispatcher makes a worker `Ready` after assigning task to it;
			// `Ready` -> `Running`:   Worker makes itself run when resume from waiting on cv
			// `Running` -> `Waiting`: Worker waits on cv if it can't find any task
			//static constexpr unsigned Waiting = 0b0001; // Waiting for task
			//static constexpr unsigned Pending = 0b0010; // Has pending task
			//static constexpr unsigned Running = 0b0100; // Executing task

			unsigned alignas(64) unsigned pending_{ 0 };
			rng_t rng_;

			// Push a forked task onto stack
			void _push(task_wrapper tw)
			{
				run_stack_.push_back(tw); // Notify one?
			}
			// Pop a task from stack for the worker itself to execute
			[[nodiscard]] bool _pop(task_wrapper& tw) noexcept
			{
				return run_stack_.pop_back(tw);
			}
			[[nodiscard]] bool _steal(task_wrapper& tw)
			{
				return etor_->steal(tw, index_, &rng_);
			}
		public:
			//static constexpr auto RUN_QUEUE_SIZE = 256u;
			worker(hb_executor& etor, std::size_t idx) :
				etor_(&etor), index_(idx) {}
			~worker() {
#ifdef PRINT_ETOR
				printf("~worker(): @%llu\n", this->index_);
#endif
			}
			worker(worker&& oth) noexcept // Should not be used, only for vector
				: etor_(std::exchange(oth.etor_, nullptr))
				, index_(std::exchange(oth.index_, -1))
				, run_stack_(std::move(oth.run_stack_))
				/*, state_(oth.state_)*/
				, rng_(std::move(oth.rng_)) {}
			worker& operator=(const worker&) = delete;

			void operator()(std::stop_token stoken)
			{
				auto h = get_handle();
				while (!etor_->is_done()) {
#ifdef PRINT_ETOR
					printf("\n@worker %llu: ", this->index_);
#endif
					// This task wrapper must be destroyed at the end of the loop
					task_wrapper tw; 
					
					// get work from local stack 
					if (_pop(tw)) {
#ifdef PRINT_ETOR
						printf("get work from local queue\n");
#endif
						tw.run(h);
						continue;
					}
					
					// steal from others
					if (etor_->steal(tw, index_, &rng_)) {
#ifdef PRINT_ETOR
						printf("steal work from others\n");
#endif
						tw.run(h);
						continue;
					}
					
					// Try to go to sleep(waiting on a cv) if no task found
					// The cv is hooked to `jthread` object so OS thread could be woken 
					// during `cv.wait()` if `stop_requested()`
					{
#ifdef PRINT_ETOR
						printf("no work, tryna sleep, lock the mutex...\n");
#endif					
						std::unique_lock lock{ mtx_ };
						// Wait for notification or stop request(interrupts)
						cv_.wait(lock, stoken, [this]() { return pending_; });

#ifdef PRINT_ETOR
						printf("@worker %llu: ...mutex relocked, wake up to see\n", this->index_);
#endif
						if (stoken.stop_requested()) {
							break;
						}
						pending_ = false;
					} // end of unique lock
				} // End of while loop
#ifdef PRINT_ETOR
				printf("\n@worker %llu: quitting...\n", this->index_);
#endif
			}
			[[nodiscard]] bool try_assign(task_wrapper& tw)
			{
				return run_stack_.try_push_front(tw);
			}
			[[nodiscard]] bool try_steal(task_wrapper& tw) noexcept
			{
				return run_stack_.try_pop_front(tw);
			}
			void notify_work() {
				{
					std::lock_guard guard{ mtx_ };
					pending_ = static_cast<unsigned>(true);
				}
				cv_.notify_one();
			}
			worker_handle get_handle() noexcept
			{
				return worker_handle{ this };
			}
		};

		// Worker thread's main function
		static void thread_main(std::stop_token stoken, hb_executor* this_, std::size_t init_idx)
		{
#ifdef PRINT_ETOR
			printf("@thread %llu: spinning up\n", init_idx);
#endif
			this_->workers_[init_idx].operator()(std::move(stoken));
#ifdef PRINT_ETOR
			printf("@thread: shutting down\n");
#endif
		}
		
		// --------------------------------------------------------------------------------
		// Data members of executor
		// --------------------------------------------------------------------------------
		mutable std::atomic<bool> is_done_{ false };
		std::atomic<size_t> ticket_{ 0 };
		std::vector<worker> workers_;
		std::vector<std::jthread> threads_;

#ifdef COUNT_STEALING
		std::atomic<size_t> alignas(64) steal_count_ { 0 };
#endif

	public:
		bool done() noexcept
		{
			if (is_done()) {
				return true;
			}

			bool done = false;
			is_done_.compare_exchange_strong(done, true, std::memory_order_acq_rel);
			return is_done();
		}
		bool is_done() const noexcept
		{
			return is_done_.load(std::memory_order_acquire);
		}
#ifdef COUNT_STEALING
		size_t get_steal_count() const noexcept {
			return steal_count_.load(std::memory_order_acquire);
		}
#endif
	private:
		// not thread-safe (single producer, multi consumers)
		std::size_t random_idx(rng_t* rng) noexcept
		{
			static thread_local std::mt19937_64 engine;
			if (rng) {
				return std::uniform_int_distribution<std::size_t>()(*rng);
			}
			else {
				return std::uniform_int_distribution<std::size_t>()(engine);
			}
		}
		void dispatch(task_wrapper tw)
		{
			auto idx = ticket_.load();
			const auto sz = workers_.size();
			for (;;) {
#ifdef PRINT_ETOR
				printf("\n@dispatcher: ");
#endif
				if (workers_[idx % sz].try_assign(tw)) {
					workers_[idx % sz].notify_work();
					ticket_.compare_exchange_strong(idx, idx + 1, std::memory_order_acq_rel);
#ifdef PRINT_ETOR
					printf("assign work to and notified @worker %llu\n", idx % sz);
#endif
					break;
				}
				else {
					++idx;
				}
				// Fall back to sleep
			}
		} 
		[[nodiscard]] bool steal(task_wrapper& tw, const std::size_t idx, rng_t* rng)
		{
#ifdef PRINT_ETOR
			printf("\n@worker %llu try to steal: ", idx);
#endif
			// Try stealing randomly 
			size_t victim_idx = random_idx(rng) % workers_.size();
			while (workers_.size() > 1 
				&& (victim_idx == idx)) {
				victim_idx = random_idx(rng) % workers_.size();
			}
			if (workers_[victim_idx].try_steal(tw)) {
#ifdef PRINT_ETOR
				printf("task stolen at first try from @worker %llu\n", victim_idx);
#endif
#ifdef COUNT_STEALING
				steal_count_.fetch_add(1, std::memory_order_acq_rel);
#endif
				return true;
			}

			// If failed to find a victim after pre-def attempts, 
			// decay to traversal
			for (auto& worker : workers_) {
				if (worker.try_steal(tw)) {
#ifdef PRINT_ETOR
					printf("task stolen from @worker %llu\n", victim_idx);
#endif
#ifdef COUNT_STEALING
				steal_count_.fetch_add(1, std::memory_order_acq_rel);
#endif
					return true;
				}
			}
			return false;
		}		
			
	public:				
		hb_executor(std::size_t parallelism = std::thread::hardware_concurrency())
		{
			workers_.reserve(parallelism);
			threads_.reserve(parallelism);
			for (auto i = 0u; i < parallelism; ++i) {
				workers_.emplace_back(*this, i);
				threads_.emplace_back(thread_main, this, i);
			}
		}
		~hb_executor()
		{
#ifdef PRINT_ETOR
			printf("\n@~executor call done(): ");
#endif
			while (!done()) {
#ifdef PRINT_ETOR
				printf("%s\n", is_done() ? "yes" : "no");
#endif
			}			
			for (auto& t : threads_) {
				t.request_stop();
			}
			
#ifdef PRINT_ETOR
			printf("\n@hb_executor: end\n");
#endif
		}

		hb_executor(const hb_executor&) = delete;
		hb_executor& operator=(const hb_executor&) = delete;

		template <typename F, typename R = std::invoke_result_t<F, worker_handle&>>
		requires std::invocable<F, hb_executor::worker_handle&>
		[[nodiscard]] future_t<R> execute_return(F&& func)
		{
			if (is_done()) {
				return future_t<R>{};
			}
			auto t = make_task<F, R>(std::forward<F>(func));
			auto fut = t.get_future();
			dispatch( std::move(t) );
			return fut;
		}
		
		// Submit a task to current thread and doesn't return result
		template <typename F>
		requires std::invocable<F, hb_executor::worker_handle&>
		void execute(F&& func)
		{
			if (is_done()) { return; }
			dispatch( std::forward<F>(func) );
		}
	};

	template <typename F>
	concept is_hb_task = std::invocable<F, hb_executor::worker_handle&>;
}

#endif