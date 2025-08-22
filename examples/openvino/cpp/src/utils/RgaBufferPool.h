//
// Created by Meonardo on 7/7/2024.
//

#ifndef GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_RGABUFFERPOOL_H_
#define GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_RGABUFFERPOOL_H_

#include <vector>

#include "im2d.h"
#include "RgaUtils.h"

#include "DmaAlloc.h"
#include "../VideoSink.h"

struct ExternalBufferGroup;

struct ExtMppBufferAllocator {
  int buf_count;
  uint32_t buf_size;

  ExtMppBufferAllocator(int count, uint32_t size);
  ~ExtMppBufferAllocator();

  void InitBuffer(int idx, rga_buffer_t* buffer) const;

 private:
  ExternalBufferGroup *buffer_group_;
};

enum BufferType {
  kBufferTypeCpu = 0,
  kBufferTypeDma = 1,
  kBufferTypeMpp = 2,
};

class RgaBufferPool {
 public:
  RgaBufferPool(size_t pool_size,
				RgaSURF_FORMAT format,
				int w = kBaseVideoWidth,
				int h = kBaseVideoHeight,
#if USE_VIRTUAL_ADDRESS
	  BufferType buf_type = kBufferTypeCpu);
#elif USE_MPP_BUFFER
				BufferType buf_type = kBufferTypeMpp);
#else
  BufferType buf_type = kBufferTypeDma);
#endif

  ~RgaBufferPool();

  // copy
  RgaBufferPool(const RgaBufferPool &);
  RgaBufferPool &operator=(const RgaBufferPool &);

  rga_buffer_t *Acquire();
  void Recycle(rga_buffer_t *buffer);

  void Refresh();

 private:
  size_t pool_size_;
  BufferType buf_type_;
  ExternalBufferGroup *buffer_group_;

  int width_;
  int height_;
  int hor_stride_;
  int ver_stride_;
  size_t rga_buffer_size_;
  RgaSURF_FORMAT format_;
  size_t current_index_;
  std::vector<std::unique_ptr<rga_buffer_t>> buffers_;

  bool ImportRgaBuffer(rga_buffer_t *buffer);
};

#endif //GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_RGABUFFERPOOL_H_
