#include "Utils.h"

#include <cmath>
#include <string>
#include <vector>

#include "Common.h"

#define TAG "Utils"

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

namespace util::rga {

int create_rga_buffer(uint32_t w, uint32_t h, RgaSURF_FORMAT fmt, rga_buffer_t& out_buffer) {
  int ret = 0;
  int fd = 0;
  void* vir_addr = nullptr;
  rga_buffer_handle_t rga_handle = 0;
  uint32_t rga_buffer_size = 0;
  uint32_t hor_stride = MPP_ALIGN(w, 16);
  uint32_t ver_stride = MPP_ALIGN(h, 16);

  if (fmt == RK_FORMAT_RGB_888) {
    rga_buffer_size = hor_stride * ver_stride * 3;
  } else if (fmt == RK_FORMAT_YCbCr_420_SP) {
    rga_buffer_size = hor_stride * ver_stride * 3 / 2;
  } else {
    assert(false && "create_rga_buffer: unsupported format");
  }

  ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, rga_buffer_size, &fd, (void**) &vir_addr);
  if (ret != 0) {
    LOGE(TAG, "Create dma buffer failed: ret: %d", ret);
    return ret;
  }

  if (vir_addr != nullptr) {
    if (fmt == RK_FORMAT_RGB_888) {
      memset(vir_addr, 0x00, rga_buffer_size);  // black
    } else if (fmt == RK_FORMAT_YCbCr_420_SP) {
      size_t y_size = hor_stride * ver_stride;
      size_t uv_size = y_size / 2;
      memset(vir_addr, 0x00, y_size);                    // Y plane black
      memset((char*) vir_addr + y_size, 0x80, uv_size);  // UV plane gray
    }
  }

  im_handle_param_t handle_param;
  handle_param.width = w;
  handle_param.height = ver_stride;
  handle_param.format = fmt;

  rga_handle = importbuffer_fd(fd, &handle_param);
  if (rga_handle == 0) {
    LOGE(TAG, "rga import dma buffer failed, fd=%d", fd);
    dma_buf_free(rga_buffer_size, &fd, vir_addr);
    return rga_handle;
  }

  out_buffer = wrapbuffer_handle(rga_handle, w, h, (int) fmt, hor_stride, ver_stride);
  out_buffer.fd = fd;              // keep the fd for later free
  out_buffer.vir_addr = vir_addr;  // keep the vir_addr for later free

  return ret;
}

void release_rga_buffer(rga_buffer_t& buffer) {
  if (buffer.handle > 0) {
    releasebuffer_handle(buffer.handle);
    buffer.handle = -1;
  }
  if (buffer.fd > 0) {
    uint32_t buf_size = 0;
    if (buffer.format == RK_FORMAT_RGB_888) {
      buf_size = buffer.wstride * buffer.hstride * 3;
    } else if (buffer.format == RK_FORMAT_YCbCr_420_SP) {
      buf_size = buffer.wstride * buffer.hstride * 3 / 2;
    }
    dma_buf_free(buf_size, &buffer.fd, buffer.vir_addr);
    buffer.fd = -1;
    buffer.vir_addr = nullptr;
  }
}

}  // namespace util::rga

namespace util::rknn {

void dump_tensor_attr(rknn_tensor_attr* attr) {
  LOGD(TAG,
       "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, "
       "type=%s, qnt_type=%s, "
       "zp=%d, scale=%f",
       attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2],
       attr->dims[3], attr->n_elems, attr->size, get_format_string(attr->fmt),
       get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

void dump_perf_info(rknn_perf_detail* perf_info) {
  std::string perf_str(perf_info->perf_data, perf_info->data_len);
  LOGI(TAG, "RKNN Performance Detail: %s", perf_str.c_str());
}

unsigned char* load_model(const char* filename, uint32_t* model_size) {
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    LOGE(TAG, "open model file %s fail", filename);
    return nullptr;
  }
  fseek(fp, 0, SEEK_END);
  auto model_len = ftell(fp);
  auto model = (unsigned char*) malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    LOGE(TAG, "read model file %s fail!", filename);

    fclose(fp);
    free(model);
    return nullptr;
  }

  *model_size = (uint32_t) model_len;
  fclose(fp);

  return model;
}

inline float fast_exp(float x) {
  // return exp(x);
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (12102203.1616540672 * x + 1064807160.56887296);
  return v.f;
}

float sigmoid(float x) {
  return 1.0 / (1.0 + fast_exp(-x));
}

float unsigmoid(float y) {
  return -1.0 * logf((1.0 / y) - 1.0);
}

inline int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t) __clip(dst_val, -128, 127);
  return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float) qnt - (float) zp) * scale;
}

void compute_dfl(float* tensor, int dfl_len, float* box) {
  for (int b = 0; b < 4; b++) {
    std::vector<float> exp_t(dfl_len);
    float exp_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i < dfl_len; i++) {
      exp_t[i] = fast_exp(tensor[i + b * dfl_len]);
      exp_sum += exp_t[i];
    }

    for (int i = 0; i < dfl_len; i++) {
      acc_sum += exp_t[i] / exp_sum * i;
    }
    box[b] = acc_sum;
  }
}

// Convert RGB24 to NHWC float32 with stride support
void rgb24_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride, float* dst,
                                float scale) {
  // stride: number of pixels per row (may be larger than W)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * stride * 3;
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t) y * W + x) * 3;
      dst[idx + 0] = static_cast<float>(row[3 * x + 0]) * scale;  // R
      dst[idx + 1] = static_cast<float>(row[3 * x + 1]) * scale;  // G
      dst[idx + 2] = static_cast<float>(row[3 * x + 2]) * scale;  // B
    }
  }
}

// Convert RGBA32 to NHWC float32 with stride support
void rgba32_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride, float* dst,
                                 float scale) {
  // stride: number of pixels per row (may be larger than W)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * stride * 4;  // 4 bytes per pixel
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t) y * W + x) * 3;
      dst[idx + 0] = static_cast<float>(row[4 * x + 0]) * scale;  // R
      dst[idx + 1] = static_cast<float>(row[4 * x + 1]) * scale;  // G
      dst[idx + 2] = static_cast<float>(row[4 * x + 2]) * scale;  // B
      // row[4 * x + 3] is Alpha, ignored
    }
  }
}

// Convert RGBA32 to NHWC float32 with stride support
void rgba32_to_nhwc_i8_stride(const uint8_t* src, int W, int H, int stride, int8_t* dst) {
  // stride: number of pixels per row (may be larger than W)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * stride * 4;  // 4 bytes per pixel
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t) y * W + x) * 3;
      dst[idx + 0] = static_cast<int8_t>(row[4 * x + 0] - 128);  // R
      dst[idx + 1] = static_cast<int8_t>(row[4 * x + 1] - 128);  // G
      dst[idx + 2] = static_cast<int8_t>(row[4 * x + 2] - 128);  // B
      // row[4 * x + 3] is Alpha, ignored
    }
  }
}

// Convert RGB24 to NHWC float32
void rgb24_to_nhwc_float(const uint8_t* src, int W, int H, float* dst, float scale) {
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * W * 3;
    for (int x = 0; x < W; ++x) {
      const size_t idx = (static_cast<size_t>(y) * W + x) * 3;
      dst[idx + 0] = row[3 * x + 0] * scale;  // R
      dst[idx + 1] = row[3 * x + 1] * scale;  // G
      dst[idx + 2] = row[3 * x + 2] * scale;  // B
    }
  }
}

}  // namespace util::rknn