#ifndef RING_BUFFER_H_
#define RING_BUFFER_H_

#include <atomic>
#include <vector>
#include <cassert>

template<typename T>
class RingBuffer {
 public:
  explicit RingBuffer(size_t capacity)
	  : capacity_(NextPowerOfTwo(capacity)),
		buffer_(std::unique_ptr<T[]>(new T[capacity_])),
		head_(0),
		tail_(0) {
	assert(capacity_ >= 2 && "Capacity must be at least 2");
  }

  bool Enqueue(const T &item) {
	size_t head = head_.load(std::memory_order_relaxed);
	size_t next_head = (head + 1) & (capacity_ - 1);

	if (next_head == tail_.load(std::memory_order_acquire)) {
	  // Buffer is full.
	  return false;
	}

	buffer_[head] = item;
	head_.store(next_head, std::memory_order_release);
	return true;
  }

  bool Enqueue(T &&item) {
	size_t head = head_.load(std::memory_order_relaxed);
	size_t next_head = (head + 1) & (capacity_ - 1);

	if (next_head == tail_.load(std::memory_order_acquire)) {
	  // Buffer is full.
	  return false;
	}

	buffer_[head] = std::move(item);
	head_.store(next_head, std::memory_order_release);
	return true;
  }

  // Removes an item from the buffer.
  // Returns false if the buffer is empty.
  bool Dequeue(T *item) {
	size_t tail = tail_.load(std::memory_order_relaxed);

	if (tail == head_.load(std::memory_order_acquire)) {
	  // Buffer is empty.
	  return false;
	}

	*item = buffer_[tail];
	tail_.store((tail + 1) & (capacity_ - 1), std::memory_order_release);
	return true;
  }

  bool IsEmpty() const {
	return head_.load(std::memory_order_acquire) ==
		tail_.load(std::memory_order_acquire);
  }

  bool IsFull() const {
	size_t next_head =
		(head_.load(std::memory_order_relaxed) + 1) & (capacity_ - 1);
	return next_head == tail_.load(std::memory_order_acquire);
  }

  // Returns the capacity of the buffer.
  size_t Capacity() const { return capacity_; }

  // Returns the number of items in the buffer.
  size_t Size() const {
	size_t head = head_.load(std::memory_order_acquire);
	size_t tail = tail_.load(std::memory_order_acquire);
	return (head + capacity_ - tail) & (capacity_ - 1);
  }

 private:
  size_t capacity_;
  std::unique_ptr<T[]> buffer_;
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;

  // Calculates the next power of two greater than or equal to n.
  static size_t NextPowerOfTwo(size_t n) {
	if (n == 0) return 1;
	size_t power = 1;
	while (power < n) power <<= 1;
	return power;
  }
};

#endif  // RING_BUFFER_H_
