#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ContextPool final {
 public:
  explicit ContextPool(int n) {
    for (int i = 0; i < n; ++i) q_.push(i);
  }

  int Acquire() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this]() { return !q_.empty(); });
    int id = q_.front();
    q_.pop();
    return id;
  }

  void Release(int id) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      q_.push(id);
    }
    cv_.notify_one();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<int> q_;
};

class ThreadPool final {
 public:
  // If max_queue_size == 0 -> unbounded queue.
  explicit ThreadPool(size_t num_threads, size_t max_queue_size = 0)
      : max_queue_size_(max_queue_size) {
    Start(num_threads);
  }

  ~ThreadPool() { Stop(); }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Fire-and-forget. Returns false if pool is stopping (task not queued).
  bool Post(std::function<void()> fn) { return Enqueue(std::move(fn)); }

  // Submit a task and get a future (like dispatch_async + group result).
  template <typename Fn, typename... Args>
  auto Submit(Fn&& fn, Args&&... args)
      -> std::future<std::invoke_result_t<Fn, Args...>> {
    using R = std::invoke_result_t<Fn, Args...>;

    auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));

    std::future<R> fut = task->get_future();
    bool ok = Enqueue([task]() { (*task)(); });
    if (!ok) {
      throw std::runtime_error("ThreadPool is stopping; Submit() rejected.");
    }
    return fut;
  }

  // Stop accepting new work, drain queued work, and join workers.
  void Stop() {
    bool expected = false;
    if (!stopping_.compare_exchange_strong(expected, true)) {
      // Already stopping/stopped.
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      // no-op; just synchronize with workers waiting on cv_.
    }
    cv_.notify_all();
    cv_space_.notify_all();

    for (std::thread& t : threads_) {
      if (t.joinable()) t.join();
    }
    threads_.clear();
  }

  bool IsStopping() const { return stopping_.load(std::memory_order_acquire); }

 private:
  void Start(size_t num_threads) {
    if (num_threads == 0) num_threads = 1;
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this]() { WorkerLoop(); });
    }
  }

  bool Enqueue(std::function<void()> fn) {
    if (!fn) return false;

    std::unique_lock<std::mutex> lock(mu_);

    // Reject if already stopping.
    if (stopping_.load(std::memory_order_acquire)) return false;

    // Bounded queue: wait until there is space or stopping.
    if (max_queue_size_ > 0) {
      cv_space_.wait(lock, [this]() {
        return stopping_.load(std::memory_order_acquire) ||
               queue_.size() < max_queue_size_;
      });
      if (stopping_.load(std::memory_order_acquire)) return false;
    }

    queue_.push_back(std::move(fn));
    cv_.notify_one();
    return true;
  }

  void WorkerLoop() {
    for (;;) {
      std::function<void()> task;

      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() {
          return stopping_.load(std::memory_order_acquire) || !queue_.empty();
        });

        if (queue_.empty()) {
          // If stopping and no work left, exit.
          if (stopping_.load(std::memory_order_acquire)) return;
          continue;
        }

        task = std::move(queue_.front());
        queue_.pop_front();
        if (max_queue_size_ > 0) cv_space_.notify_one();
      }

      // Run outside lock.
      task();
    }
  }

  const size_t max_queue_size_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable cv_space_;
  std::deque<std::function<void()>> queue_;

  std::vector<std::thread> threads_;
  std::atomic<bool> stopping_{false};
};

#endif  // THREAD_POOL_H