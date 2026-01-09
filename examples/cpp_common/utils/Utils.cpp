#include "Utils.h"

#include <cmath>
#include <string>
#include <vector>

#include "Common.h"

#define TAG "Utils"

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

namespace util::rga {

int create_rga_buffer(uint32_t w, uint32_t h, RgaSURF_FORMAT fmt,
                      rga_buffer_t& out_buffer) {
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

  ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, rga_buffer_size, &fd,
                      (void**)&vir_addr);
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
      memset(vir_addr, 0x00, y_size);                   // Y plane black
      memset((char*)vir_addr + y_size, 0x80, uv_size);  // UV plane gray
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

  out_buffer =
      wrapbuffer_handle(rga_handle, w, h, (int)fmt, hor_stride, ver_stride);
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
       "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, "
       "size=%d, fmt=%s, "
       "type=%s, qnt_type=%s, "
       "zp=%d, scale=%f",
       attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
       attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
       get_format_string(attr->fmt), get_type_string(attr->type),
       get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
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
  auto model = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    LOGE(TAG, "read model file %s fail!", filename);

    fclose(fp);
    free(model);
    return nullptr;
  }

  *model_size = (uint32_t)model_len;
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

float sigmoid(float x) { return 1.0 / (1.0 + fast_exp(-x)); }

float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
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
void rgb24_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride,
                                float* dst, float scale) {
  // stride: number of pixels per row (may be larger than W)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * stride * 3;
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t)y * W + x) * 3;
      dst[idx + 0] = static_cast<float>(row[3 * x + 0]) * scale;  // R
      dst[idx + 1] = static_cast<float>(row[3 * x + 1]) * scale;  // G
      dst[idx + 2] = static_cast<float>(row[3 * x + 2]) * scale;  // B
    }
  }
}

// Convert RGBA32 to NHWC float32 with stride support
void rgba32_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride,
                                 float* dst, float scale) {
  // stride: number of pixels per row (may be larger than W)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row =
        src + static_cast<size_t>(y) * stride * 4;  // 4 bytes per pixel
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t)y * W + x) * 3;
      dst[idx + 0] = static_cast<float>(row[4 * x + 0]) * scale;  // R
      dst[idx + 1] = static_cast<float>(row[4 * x + 1]) * scale;  // G
      dst[idx + 2] = static_cast<float>(row[4 * x + 2]) * scale;  // B
      // row[4 * x + 3] is Alpha, ignored
    }
  }
}

// Convert RGBA32 to NHWC float32 with stride support
void rgba32_to_nhwc_i8_stride(const uint8_t* src, int W, int H, int stride,
                              int8_t* dst) {
  // stride: number of pixels per row (may be larger than W)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row =
        src + static_cast<size_t>(y) * stride * 4;  // 4 bytes per pixel
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t)y * W + x) * 3;
      dst[idx + 0] = static_cast<int8_t>(row[4 * x + 0] - 128);  // R
      dst[idx + 1] = static_cast<int8_t>(row[4 * x + 1] - 128);  // G
      dst[idx + 2] = static_cast<int8_t>(row[4 * x + 2] - 128);  // B
      // row[4 * x + 3] is Alpha, ignored
    }
  }
}

// Convert RGB24 to NHWC float32
void rgb24_to_nhwc_float(const uint8_t* src, int W, int H, float* dst,
                         float scale) {
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

namespace face {

// similarity transform destination points
static const std::vector<cv::Point2f> SIMILARITY_TRANSFORM_DEST = {
    {38.2946, 51.6963},
    {73.5318, 51.5014},
    {56.0252, 71.7366},
    {41.5493, 92.3655},
    {70.7299, 92.2041}};

static cv::Mat compute_similarity2d(const std::vector<cv::Point2f>& src,
                                    const std::vector<cv::Point2f>& dst) {
  CV_Assert(src.size() == dst.size());
  CV_Assert(src.size() >= 2);

  const int n = static_cast<int>(src.size());

  // Compute means.
  cv::Point2d mu_src(0.0, 0.0), mu_dst(0.0, 0.0);
  for (int i = 0; i < n; ++i) {
    mu_src.x += src[i].x;
    mu_src.y += src[i].y;
    mu_dst.x += dst[i].x;
    mu_dst.y += dst[i].y;
  }
  mu_src.x /= n;
  mu_src.y /= n;
  mu_dst.x /= n;
  mu_dst.y /= n;

  // Compute centered coordinates and variance of src.
  double var_src = 0.0;
  cv::Matx<double, 2, 2> cov(0.0, 0.0, 0.0, 0.0);  // covariance src->dst
  for (int i = 0; i < n; ++i) {
    const double xs = src[i].x - mu_src.x;
    const double ys = src[i].y - mu_src.y;
    const double xd = dst[i].x - mu_dst.x;
    const double yd = dst[i].y - mu_dst.y;

    var_src += xs * xs + ys * ys;

    // cov += [xd; yd] * [xs ys]
    cov(0, 0) += xd * xs;
    cov(0, 1) += xd * ys;
    cov(1, 0) += yd * xs;
    cov(1, 1) += yd * ys;
  }

  if (var_src <= 1e-12) {
    // Degenerate: all src points identical.
    return cv::Mat::eye(2, 3, CV_64F);
  }

  cov *= (1.0 / n);
  var_src *= (1.0 / n);

  // SVD of covariance - must use cv::Mat for output
  cv::Mat U, Vt, S_vec;
  cv::SVD::compute(cv::Mat(cov), S_vec, U, Vt);

  // Extract singular values
  cv::Vec<double, 2> S(S_vec.at<double>(0), S_vec.at<double>(1));

  // Convert U and Vt to Matx for easier math
  cv::Matx<double, 2, 2> U_matx(U.at<double>(0, 0), U.at<double>(0, 1),
                                U.at<double>(1, 0), U.at<double>(1, 1));
  cv::Matx<double, 2, 2> Vt_matx(Vt.at<double>(0, 0), Vt.at<double>(0, 1),
                                 Vt.at<double>(1, 0), Vt.at<double>(1, 1));

  // Rotation R = U * Vt (with reflection fix).
  cv::Matx<double, 2, 2> R = U_matx * Vt_matx;

  // If det(R) < 0, fix reflection by flipping the last column of U.
  const double detR = R(0, 0) * R(1, 1) - R(0, 1) * R(1, 0);
  if (detR < 0.0) {
    U_matx(0, 1) *= -1.0;
    U_matx(1, 1) *= -1.0;
    R = U_matx * Vt_matx;
    S[1] *= -1.0;  // keep trace consistent
  }

  // Scale: s = trace(S) / var_src
  const double scale = (S[0] + S[1]) / var_src;

  // Translation: t = mu_dst - s * R * mu_src
  const cv::Vec2d muS(mu_src.x, mu_src.y);
  const cv::Vec2d muD(mu_dst.x, mu_dst.y);
  const cv::Vec2d t = muD - scale * (R * muS);

  cv::Mat M(2, 3, CV_64F);
  M.at<double>(0, 0) = scale * R(0, 0);
  M.at<double>(0, 1) = scale * R(0, 1);
  M.at<double>(1, 0) = scale * R(1, 0);
  M.at<double>(1, 1) = scale * R(1, 1);
  M.at<double>(0, 2) = t[0];
  M.at<double>(1, 2) = t[1];

  return M;
}

cv::Mat affine_crop(cv::Mat& img, float kps[10], int size, cv::Mat& dst_mat) {
  std::vector<cv::Point2f> points_five;
  for (size_t i = 0; i < 5; i++) {
    points_five.push_back(cv::Point2f(kps[i * 2], kps[i * 2 + 1]));
  }

  std::vector<cv::Point2f> dst(SIMILARITY_TRANSFORM_DEST);
  float ratio = float(size) / 112.0;
  float diff_x = 0.f;
  for (auto& d : dst) {
    d.x = d.x * ratio;
    d.y = d.y * ratio;
  }
  for (auto& d : dst) {
    d.x = d.x + diff_x;
  }

  cv::Mat M = compute_similarity2d(points_five, dst);
  cv::warpAffine(img, dst_mat, M, cv::Size(size, size), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  return M;
}

cv::Rect2f make_square_crop(float x, float y, float w, float h, float img_w,
                            float img_h, float scale /* 1.2f */) {
  float cx = x + 0.5f * w;
  float cy = y + 0.5f * h;
  float side = std::max(w, h) * scale;

  float sx = cx - 0.5f * side;
  float sy = cy - 0.5f * side;

  // Clamp to image bounds
  if (sx < 0) sx = 0;
  if (sy < 0) sy = 0;
  if (sx + side > img_w) sx = img_w - side;
  if (sy + side > img_h) sy = img_h - side;
  if (sx < 0) sx = 0;
  if (sy < 0) sy = 0;

  return cv::Rect2f{sx, sy, side, side};
}

cv::Point2f landmark_to_patch_112(const cv::Point2f& p_img,
                                  const cv::Rect2f& crop /* in img space */,
                                  float out_size /*112*/) {
  float sx = out_size / crop.width;
  float sy = out_size / crop.height;
  return cv::Point2f((p_img.x - crop.x) * sx, (p_img.y - crop.y) * sy);
}

}  // namespace face