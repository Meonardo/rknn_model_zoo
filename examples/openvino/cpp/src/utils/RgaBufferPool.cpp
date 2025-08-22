//
// Created by Meonardo on 7/7/2024.
//

#include "RgaBufferPool.h"

#include "mpp_mem.h"
#include "mpp_env.h"
#include "mpp_time.h"
#include "mpp_common.h"

#include "../Common.h"
#include "../VideoSink.h"

#define TAG "RgaBufferPool"

struct ExternalBufferGroup {
  RK_U32 buf_count;
  RK_U32 buf_size;

  MppBufferGroup group;
  MppBuffer *buf_list;

  ExternalBufferGroup(RK_U32 count, RK_U32 size) : buf_count(count),
												   buf_size(size),
												   group(nullptr),
												   buf_list(nullptr) {
	MPP_RET ret = MPP_NOK;
	ret = mpp_buffer_group_get_external(&group, MPP_BUFFER_TYPE_ION);
	if (ret) {
	  LOGE(TAG, "get mpp external buffer group failed ret %d", ret);
	  return;
	}
	buf_list = mpp_calloc(MppBuffer, count);
	if (buf_list == nullptr) {
	  LOGE(TAG, "malloc buffer list failed");
	  return;
	}
  }

  void CommitBuffer(int idx, rga_buffer_t* buffer) const {
	if (idx >= buf_count) {
	  LOGE(TAG, "invalid buffer index %d\n", idx);
	  return;
	}

	MppBufferInfo commit;
	commit.type = MPP_BUFFER_TYPE_ION;
	commit.size = buf_size;

	auto ret = mpp_buffer_get(nullptr, &buf_list[idx], buf_size);
	if (ret || nullptr == buf_list[idx]) {
	  LOGE(TAG, "get misc buffer failed ret %d", ret);
	  return;
	}

	commit.index = idx;
	commit.ptr = mpp_buffer_get_ptr(buf_list[idx]);
	commit.fd = mpp_buffer_get_fd(buf_list[idx]);

	ret = mpp_buffer_commit(group, &commit);
	if (ret) {
	  LOGE(TAG, "external buffer commit failed ret %d", ret);
	  return;
	}

	// save the buffer info
	buffer->fd = mpp_buffer_get_fd(buf_list[idx]);
	buffer->vir_addr = mpp_buffer_get_ptr(buf_list[idx]);
  }

  ~ExternalBufferGroup() {
	if (group != nullptr) {
	  mpp_buffer_group_put(group);
	  group = nullptr;
	}

	if (buf_list != nullptr) {
	  RK_U32 i;

	  for (i = 0; i < buf_count; i++) {
		if (buf_list[i]) {
		  mpp_buffer_put(buf_list[i]);
		  buf_list[i] = nullptr;
		}
	  }

	  MPP_FREE(buf_list);
	}
  }
};

ExtMppBufferAllocator::ExtMppBufferAllocator(int count, uint32_t size)
	: buf_count(count),
	  buf_size(size),
	  buffer_group_(new ExternalBufferGroup(count, size)) {
}

ExtMppBufferAllocator::~ExtMppBufferAllocator() {
  delete buffer_group_;
  buffer_group_ = nullptr;
}

void ExtMppBufferAllocator::InitBuffer(int idx, rga_buffer_t *buffer) const {
  buffer_group_->CommitBuffer(idx, buffer);
}

RgaBufferPool::RgaBufferPool(size_t pool_size,
							 RgaSURF_FORMAT format,
							 int w,
							 int h,
							 BufferType buf_type)
	: pool_size_(pool_size),
	  buf_type_(buf_type),
	  buffer_group_(nullptr),
	  width_(w),
	  height_(h),
	  hor_stride_(MPP_ALIGN(w, 16)),
#if USE_VIRTUAL_ADDRESS
	ver_stride_(h),
#else
	  ver_stride_(MPP_ALIGN(h, 16)),
#endif
	  rga_buffer_size_(0),
	  format_(format),
	  current_index_(0) {
  for (size_t i = 0; i < pool_size_; ++i) {
	auto buffer = std::make_unique<rga_buffer_t>();
	// init & import to RGA buffer
	if (ImportRgaBuffer(buffer.get())) {
	  // save the buffer
	  buffers_.emplace_back(std::move(buffer));
	}
  }
}

RgaBufferPool::~RgaBufferPool() {
  for (auto &buffer : buffers_) {
	if (buffer->handle > 0) {
	  // release RGA buffer
	  releasebuffer_handle(buffer->handle);
	}
	if (buffer->vir_addr != nullptr) {
	  if (buf_type_ == kBufferTypeCpu) {
		free(buffer->vir_addr);
	  } else if (buf_type_ == kBufferTypeDma) {
		// release dma buffer
		dma_buf_free(rga_buffer_size_, &buffer->fd, buffer->vir_addr);
	  }
	}
	buffer.reset();
  }

  if (buf_type_ == kBufferTypeMpp) {
	if (buffer_group_ != nullptr) {
	  delete buffer_group_;
	  buffer_group_ = nullptr;
	}
  }

  buffers_.clear();
}

bool RgaBufferPool::ImportRgaBuffer(rga_buffer_t *buffer) {
  int fd = 0;
  void *vir_addr = nullptr;
  int ret = -1;
  rga_buffer_handle_t rga_handle = 0;

  // create dma buffer
  rga_buffer_size_ = (size_t)get_bpp_from_format(format_) * width_ * height_;
  if (format_ == RK_FORMAT_YCbCr_420_SP) {
	rga_buffer_size_ = hor_stride_ * ver_stride_ * 3 / 2; // NV12
  } else if (format_ == RK_FORMAT_YCbCr_400) {
	rga_buffer_size_ = hor_stride_ * ver_stride_;
  }
  if (buf_type_ == kBufferTypeCpu) {
	vir_addr = malloc(rga_buffer_size_);
  } else if (buf_type_ == kBufferTypeDma) {
	ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH,
						rga_buffer_size_,
						&fd,
						(void **)&vir_addr);
	if (ret != 0) {
	  LOGE(TAG, "alloc dma32_heap buffer failed");
	  return false;
	}
  } else {
	if (buffer_group_ == nullptr) {
	  buffer_group_ = new ExternalBufferGroup(pool_size_, rga_buffer_size_);
	}
	buffer_group_->CommitBuffer((int)buffers_.size(), buffer);
	fd = buffer->fd;
	vir_addr = buffer->vir_addr;
  }

  if (vir_addr != nullptr) {
	// fill the buffer with black
	if (format_ == RK_FORMAT_YCbCr_420_SP) {
	  size_t y_size = hor_stride_ * ver_stride_;
	  size_t uv_size = y_size / 2;
	  memset(vir_addr, 0x00, y_size);
	  memset((char *)vir_addr + y_size, 0x80, uv_size);
	} else {
	  memset(vir_addr, 0x00, rga_buffer_size_);
	}
  }

  // import to RGA buffer
  im_handle_param_t handle_param;
  handle_param.width = width_;
  handle_param.height = ver_stride_;
  handle_param.format = format_;

  if (buf_type_ == kBufferTypeCpu) {
	rga_handle = importbuffer_virtualaddr(vir_addr, &handle_param);
	if (rga_handle <= 0) {
	  LOGE(TAG, "rga import virtual address buffer failed, address=%p", vir_addr);
	  free(vir_addr);
	  return false;
	}
  } else if (buf_type_ == kBufferTypeDma) {
	rga_handle = importbuffer_fd(fd, &handle_param);
	if (rga_handle <= 0) {
	  LOGE(TAG, "rga import dma buffer failed, fd=%d", fd);
	  dma_buf_free(rga_buffer_size_, &fd, vir_addr);
	  return false;
	}
  } else if (buf_type_ == kBufferTypeMpp) {
	rga_handle = importbuffer_fd(fd, &handle_param);
	if (rga_handle <= 0) {
	  LOGE(TAG, "rga import dma buffer failed, fd=%d", fd);
	  return false;
	}
  }

  // save the buffer info
  *buffer = wrapbuffer_handle(rga_handle,
							  width_,
							  height_,
							  (int)handle_param.format,
							  hor_stride_,
							  ver_stride_);
  buffer->fd = fd;
  buffer->vir_addr = vir_addr;

  return true;
}

RgaBufferPool::RgaBufferPool(const RgaBufferPool &other) {
  pool_size_ = other.pool_size_;
  rga_buffer_size_ = other.rga_buffer_size_;
  width_ = other.width_;
  height_ = other.height_;
  hor_stride_ = other.hor_stride_;
  ver_stride_ = other.ver_stride_;
  format_ = other.format_;
  current_index_ = 0;
  buf_type_ = other.buf_type_;

  // TODO: re-create the buffer group
  buffer_group_ = nullptr;

  for (size_t i = 0; i < pool_size_; ++i) {
	auto buffer = std::make_unique<rga_buffer_t>();
	auto old = other.buffers_[i].get();
	// copy already imported buffer
	*buffer = *old;
	// save the buffer
	buffers_.emplace_back(std::move(buffer));
  }
}

RgaBufferPool &RgaBufferPool::operator=(const RgaBufferPool &other) {
  if (this == &other) {
	return *this;
  }

  pool_size_ = other.pool_size_;
  rga_buffer_size_ = other.rga_buffer_size_;
  width_ = other.width_;
  height_ = other.height_;
  hor_stride_ = other.hor_stride_;
  ver_stride_ = other.ver_stride_;
  format_ = other.format_;
  current_index_ = 0;

  // TODO: re-create the buffer group
  buffer_group_ = nullptr;

  for (size_t i = 0; i < pool_size_; ++i) {
	auto buffer = std::make_unique<rga_buffer_t>();
	auto old = other.buffers_[i].get();
	// copy already imported buffer
	*buffer = *old;
	// save the buffer
	buffers_.emplace_back(std::move(buffer));
  }

  return *this;
}

rga_buffer_t *RgaBufferPool::Acquire() {
  auto &buffer = buffers_[current_index_];
  current_index_ = (current_index_ + 1) % buffers_.size();
  return buffer.get();
}

void RgaBufferPool::Recycle(rga_buffer_t *buffer) {
  if (buffer == nullptr) {
	return;
  }
  // move the buffer to the front
  current_index_ = (current_index_ - 1) % buffers_.size();
}

void RgaBufferPool::Refresh() {
  for (auto &buffer : buffers_) {
	// fill the buffer with black
	if (format_ == RK_FORMAT_RGBA_8888) {
	  memset(buffer->vir_addr, 0x00, rga_buffer_size_);
	} else if (format_ == RK_FORMAT_YCbCr_420_SP) {
	  size_t y_size = kBaseVideoWidth * kBaseVerticalStride;
	  size_t uv_size = y_size / 2;
	  memset(buffer->vir_addr, 0x00, y_size);
	  memset((char *)buffer->vir_addr + y_size, 0x80, uv_size);
	}
  }
}
