//
// Created by Meonardo on 7/7/2024.
//

#ifndef GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_UTILS_MPPDECODER_H_
#define GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_UTILS_MPPDECODER_H_

#include <cstring>
#include <memory>
#include <atomic>
#include <thread>
#include <queue>

#include "rk_mpi.h"
#include "mpp_frame.h"

#include "../VideoSink.h"
#include "RingBuffer.h"

namespace v_dec {
typedef void(*MppDecoderMtFrameCallback)(void *userdata,
										 RK_U32 width_stride,
										 RK_U32 height_stride,
										 RK_U32 width,
										 RK_U32 height,
										 MppFrameFormat format,
										 int fd);

typedef void *DecBufMgr;

enum MppDecBufMode {
  MPP_DEC_BUF_HALF_INT,
  MPP_DEC_BUF_INTERNAL,
  MPP_DEC_BUF_EXTERNAL,
  MPP_DEC_BUF_MODE_BUTT,
};

class BufferPool {
 public:
  explicit BufferPool(size_t pool_size, size_t max_buffer_size)
	  : pool_size_(pool_size), max_buffer_size_(max_buffer_size) {
	for (size_t i = 0; i < pool_size_; ++i) {
	  auto data = (char *)malloc(max_buffer_size_);
	  auto packet = new VideoBuffSlot();
	  if (data && packet) {
		packet->data = data;
		packet->size = (int)max_buffer_size_;
		pool_.push(packet);
	  } else {
		if (data) {
		  free(data);
		}
		delete packet;
	  }
	}
  }

  ~BufferPool() {
	std::lock_guard<std::mutex> lock(mutex_);
	while (!pool_.empty()) {
	  auto packet = pool_.front();
	  free(packet->data);
	  pool_.pop();
	}
  }

  VideoBuffSlot* Acquire(size_t size) {
	std::unique_lock<std::mutex> lock(mutex_);
	if (size > max_buffer_size_) {
	  // Allocate a packet for large size dynamically
	  auto packet = new VideoBuffSlot();
	  packet->data = (char *)malloc(size);
	  packet->size = (int)size;
	  return packet;
	} else {
	  cv_.wait(lock, [this]() { return !pool_.empty(); });
	  auto packet = pool_.front();
	  pool_.pop();
	  packet->size = (int)size;
	  return packet;
	}
  }

  void Release(VideoBuffSlot* packet) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (packet->size > max_buffer_size_) {
	  // Free dynamically allocated packet
	  free(packet->data);
	  delete packet;
	} else {
	  pool_.push(packet);
	  cv_.notify_one();
	}
  }

 private:
  size_t pool_size_;
  size_t max_buffer_size_;
  std::queue<VideoBuffSlot*> pool_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

class MppDecoder {
 public:
  MppDecoder(MppCodingType type, RK_U32 width, RK_U32 height, void *user_data);
  ~MppDecoder();

  void PutPacket(const void *data, size_t size, RK_U32 eos = 0);

  void SetCallback(MppDecoderMtFrameCallback callback) {
	callback_ = callback;
  }

 private:
  bool use_stream_buffer_;
  MppCodingType type_;
  MppFrameFormat format_;
  RK_U32 width_;
  RK_U32 height_;

  MppDecBufMode buf_mode_;

  MppCtx ctx_;
  MppApi *mpi_;

  std::atomic<bool> loop_end_;
  /* input and output */
  DecBufMgr buf_mgr_;
  MppBufferGroup frm_grp_;
  MppPacket packet_;
  size_t packet_size_;
  MppFrame frame_;
  RK_S32 frame_count_;

  // stream data
  std::unique_ptr<RingBuffer<VideoBuffSlot*>> encoded_buffer_;
  std::unique_ptr<v_dec::BufferPool> buffer_pool_;
  // decoding threads
  std::unique_ptr<std::thread> dec_input_thread_;
  std::unique_ptr<std::thread> dec_output_thread_;

  // callback
  void *user_data_;
  MppDecoderMtFrameCallback callback_;

  void InitDecoderData();
  void DeInitDecoderData();

  void DecodeNow(const void *data, size_t size, RK_U32 eos);
  // decoding threads
  void Feed();
  void Decode();

  bool CommitBufferGroup(RK_U32 w, RK_U32 h, size_t size);
};
} // namespace v_dec
#endif //GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_UTILS_MPPDECODER_H_
