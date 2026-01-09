#ifndef UTILS_H_
#define UTILS_H_

#include <array>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "DmaAlloc.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "im2d_type.h"
#include "rknn_api.h"

namespace face {

using Embeddings = std::vector<float>;

struct FaceLocation {
  cv::Rect box;
  float lmk[10];
  float score;
  float recog_score;
  int id;
  char name[64];
  Embeddings embedding;
};

struct QntParam {
  int32_t zero_point;
  float scale;
};

cv::Mat affine_crop(cv::Mat& img, float kps[10], int size, cv::Mat& dst_mat);

cv::Rect2f make_square_crop(float x, float y, float w, float h, float img_w,
                            float img_h, float scale /* 1.2f */);

cv::Point2f landmark_to_patch_112(const cv::Point2f& p_img,
                                  const cv::Rect2f& crop /* in img space */,
                                  float out_size /*112*/);

}  // namespace face

namespace util {

namespace rga {

int create_rga_buffer(uint32_t w, uint32_t h, RgaSURF_FORMAT fmt,
                      rga_buffer_t& out_buffer);
void release_rga_buffer(rga_buffer_t& buffer);

}  // namespace rga

namespace rknn {

void dump_tensor_attr(rknn_tensor_attr* attr);
void dump_perf_info(rknn_perf_detail* perf_info);
unsigned char* load_model(const char* filename, uint32_t* model_size);

float sigmoid(float x);
float unsigmoid(float y);

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

void compute_dfl(float* tensor, int dfl_len, float* box);

void rgb24_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride,
                                float* dst, float scale);
void rgba32_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride,
                                 float* dst, float scale);
void rgba32_to_nhwc_i8_stride(const uint8_t* src, int W, int H, int stride,
                              int8_t* dst);
void rgb24_to_nhwc_float(const uint8_t* src, int W, int H, float* dst,
                         float scale);

}  // namespace rknn

}  // namespace util

#endif  // UTILS_H_