#include "FaceDetector.h"

#include <array>
#include <chrono>
#include <thread>

#include "Common.h"

#define TAG "FaceDetector"

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

#define ENABLE_PROFILE 0

#if ENABLE_PROFILE
static uint32_t infer_count = 0;
static int64_t infer_npu_time = 0;
#endif

namespace face {

const uint32_t kDefaultWaitingTimeInMs = 33;
constexpr uint32_t kMaxValidBBoxes = 32;
constexpr std::array<int, 3> kAnchorStrides = {8, 16, 32};

static void generate_anchors(int stride, int input_size, int num_anchors,
                             std::vector<float>& anchors) {
  int height = ceil(input_size / stride);
  int width = ceil(input_size / stride);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      for (int k = 0; k < num_anchors; ++k) {
        anchors.push_back(i * stride);
        anchors.push_back(j * stride);
      }
    }
  }
}

FaceDetector::FaceDetector(const std::string& model_path, float det_threshold, float nms_threshold)
    : model_path_(model_path),
      initialized_(false),
      det_threshold_(det_threshold),
      nms_threshold_(nms_threshold),
      rknn_ctx_(0),
      model_input_width_(640),
      model_input_height_(640),
      frame_width_(1920),
      frame_height_(1080) {
  LOGI(TAG, "Create FaceDetector with model path: %s", model_path_.c_str());
  auto ret = Init();
  assert(ret == 0 && "FaceDetector Init failed");
  initialized_ = true;
}

FaceDetector::~FaceDetector() {
  LOGI(TAG, "Destroying FaceDetector with model path: %s", model_path_.c_str());

  DeInit();

  LOGI(TAG, "FaceDetector with model path: %s destroyed", model_path_.c_str());
}

int FaceDetector::Init() {
  int ret = 0;

  // Load model
  uint32_t model_len = 0;
  auto model = util::rknn::load_model(model_path_.c_str(), &model_len);
  if (model == nullptr) {
    LOGE(TAG, "load model failed");
    return -1;
  }

  // Init RKNN
  uint32_t flags = 0;
#if ENABLE_PROFILE
  flags |= RKNN_FLAG_COLLECT_PERF_MASK;
#endif
  ret = rknn_init(&rknn_ctx_, model, model_len, flags, nullptr);
  if (ret < 0) {
    LOGE(TAG, "rknn_init fail ret=%d", ret);

    free(model);
    return -1;
  }

  // Query IO number
  ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(rknn_input_output_num));
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "rknn_query fail ret=%d", ret);
    return -1;
  }
  LOGI(TAG, "rknn_query input num: %d, output num: %d", io_num_.n_input, io_num_.n_output);

  // Query input tensor shape
  input_attrs_.resize(io_num_.n_input);
  for (int i = 0; i < io_num_.n_input; i++) {
    input_attrs_[i].index = i;  // Set index for input tensor
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_query input attr fail ret=%d", ret);
      return -1;
    }

    util::rknn::dump_tensor_attr(&input_attrs_[i]);
  }

  // Query output tensor shape
  output_attrs_.resize(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; i++) {
    output_attrs_[i].index = i;  // Set index for output tensor
    ret =
        rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_query output attr fail ret=%d", ret);
      return -1;
    }

    util::rknn::dump_tensor_attr(&output_attrs_[i]);
  }

  // Create input tensors
  input_mems_.reserve(io_num_.n_input);
  for (int i = 0; i < io_num_.n_input; i++) {
    // Create memory for this tensor
    rknn_tensor_mem* mem = nullptr;
    if (input_attrs_[i].type == RKNN_TENSOR_FLOAT16) {
      input_attrs_[i].type = RKNN_TENSOR_FLOAT32;           // change to float32
      auto size = input_attrs_[i].n_elems * sizeof(float);  // float32
      input_attrs_[i].size = size;
      input_attrs_[i].size_with_stride = size;
      mem = rknn_create_mem(rknn_ctx_, size);
    } else {
      // default input type is int8 (normalize and quantize need compute in outside)
      // if set uint8, will fuse normalize and quantize to npu
      input_attrs_[i].type = RKNN_TENSOR_UINT8;
      mem = rknn_create_mem(rknn_ctx_, input_attrs_[i].size_with_stride);
    }
    if (!mem) {
      LOGE(TAG, "create input mem fail");
      return -1;
    }
    input_mems_.push_back(mem);

    // Set input tensor memory with attributes
    ret = rknn_set_io_mem(rknn_ctx_, mem, &input_attrs_[i]);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_set_io_mem fail ret=%d", ret);
      return ret;
    }
  }

  // Create output tensors:
  output_mems_.reserve(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; i++) {
    // Create memory for this tensor
    rknn_tensor_mem* mem = nullptr;
    if (output_attrs_[i].type == RKNN_TENSOR_FLOAT16) {
      output_attrs_[i].type = RKNN_TENSOR_FLOAT32;           // change to float32
      auto size = output_attrs_[i].n_elems * sizeof(float);  // float32
      output_attrs_[i].size = size;
      output_attrs_[i].size_with_stride = size;
      mem = rknn_create_mem(rknn_ctx_, size);
    } else {
      if (std::fabs(output_attrs_[i].scale) < std::numeric_limits<float>::epsilon()) {
        LOGE(TAG, "scale can't be equal to 0");
        output_attrs_[i].scale = 1.0;  // set a small value
      }
      mem = rknn_create_mem(rknn_ctx_, output_attrs_[i].size_with_stride);
    }
    if (!mem) {
      LOGE(TAG, "create output mem fail");
      return -1;
    }
    output_mems_.push_back(mem);

    // Set output tensor memory with attributes
    ret = rknn_set_io_mem(rknn_ctx_, mem, &output_attrs_[i]);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_set_io_mem fail ret=%d", ret);
      return ret;
    }
  }

  // Update model input width and height
  if (input_attrs_.size() > 0) {  // rknn use NHWC as default
    model_input_width_ = static_cast<uint32_t>(input_attrs_[0].dims[2]);
    model_input_height_ = static_cast<uint32_t>(input_attrs_[0].dims[1]);
    LOGI(TAG, "Model input width: %u, height: %u", model_input_width_, model_input_height_);
  }

  // Create rga buffers
  CreateRgaBuffers();

#if USE_QUANTIZED_MODEL
  // Prepare dequantization parameters for quantized model
  deqnt_params_.reserve(3);
  for (int i = 0; i < 3; i++) {
    std::vector<QntParam> p;
    p.reserve(3);
    for (int j = 0; j < 3; j++) {
      int idx = j * 3 + i;
      auto& attr = output_attrs_[idx];
      QntParam param{attr.zp, attr.scale};
      p.emplace_back(std::move(param));
    }

    deqnt_params_.emplace_back(p);
  }
#endif
  // Pre-generate anchors for each stride and cache them to avoid repeated allocations
  for (int i = 0; i < 3; ++i) {
    anchors_cache_[i].clear();
    generate_anchors(kAnchorStrides[i], model_input_width_, 2, anchors_cache_[i]);
  }

  return 0;
}

void FaceDetector::DeInit() {
  // Release RGA buffers
  DestroyRgaBuffers();

  // Free input and output memory
  for (auto* mem : input_mems_) {
    if (mem) {
      rknn_destroy_mem(rknn_ctx_, mem);
    }
  }
  for (auto* mem : output_mems_) {
    if (mem) {
      rknn_destroy_mem(rknn_ctx_, mem);
    }
  }

  // Deinitialize RKNN context
  rknn_destroy(rknn_ctx_);

  LOGI(TAG, "FaceDetector deinitialized successfully");
}

void FaceDetector::CreateRgaBuffers() {
  int ret = 0;

#if USE_QUANTIZED_MODEL
  // Try to import input tensor memory into RGA
  int fd = input_mems_[0]->fd;
  input_tensor_rga_buffer_.handle = -1;
  if (fd != 0) {
    LOGD(TAG, "Importing input tensor memory into RGA buffer");
    im_handle_param_t handle_param{};
    handle_param.width = model_input_width_;
    handle_param.height = MPP_ALIGN(model_input_height_, 16);
    handle_param.format = RK_FORMAT_RGB_888;
    rga_buffer_handle_t rga_handle = importbuffer_fd(fd, &handle_param);
    if (rga_handle > 0) {
      input_tensor_rga_buffer_ = wrapbuffer_handle(
          rga_handle, model_input_width_, model_input_height_, (int) handle_param.format,
          MPP_ALIGN(model_input_width_, 16), MPP_ALIGN(model_input_height_, 16));
      LOGD(TAG, "Input tensor memory imported");
    }
  }
#endif
  if (input_tensor_rga_buffer_.handle <= 0) {
    // fallback to DMA buffer
    ret = util::rga::create_rga_buffer(model_input_width_, model_input_height_, RK_FORMAT_RGB_888,
                                       input_tensor_rga_buffer_);
    if (ret != 0) {
      LOGE(TAG, "create scale buffer(for input tensor memory) failed: %d", ret);
      return;
    }
  }
}

void FaceDetector::DestroyRgaBuffers() {
  // Release the input tensor RGA buffer
  if (input_tensor_rga_buffer_.handle > 0) {
    releasebuffer_handle(input_tensor_rga_buffer_.handle);
    input_tensor_rga_buffer_.handle = -1;
  }
}

void FaceDetector::CalculateScale(rga_buffer_t* input_buffer) {
  frame_width_ = input_buffer->width;
  frame_height_ = input_buffer->height;

  auto original_width = (float) frame_width_;
  auto original_height = (float) frame_height_;
  // calculate the scale
  float scale = std::min((float) model_input_width_ / original_width,
                         (float) model_input_height_ / original_height);

  letter_box_.x_pad = 0;
  letter_box_.y_pad = 0;
  letter_box_.scale = scale;
}

rga_buffer_t* FaceDetector::Scale(rga_buffer_t* src) {
  if (input_tensor_rga_buffer_.handle == 0) {
    LOGE(TAG, "failed to acquire scale rga buffer");
    return nullptr;
  }

  auto width = (float) src->width * letter_box_.scale;
  auto height = (float) src->height * letter_box_.scale;

  im_rect rect = {letter_box_.x_pad, letter_box_.y_pad, (int) width, (int) height};

  auto ret = imcheck(*src, input_tensor_rga_buffer_, {}, rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "letterbox with rga check failed, %s", imStrError(ret));
    return nullptr;
  }

  ret = improcess(*src, input_tensor_rga_buffer_, {}, {}, rect, {}, IM_SYNC);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "letterbox with rga failed, %s", imStrError(ret));
    return nullptr;
  }

  return &input_tensor_rga_buffer_;
}

cv::Rect FaceDetector::UnScale(float x0, float y0, float x1, float y1) {
  // Map to original space
  x0 = (x0 - letter_box_.x_pad) / letter_box_.scale;
  y0 = (y0 - letter_box_.y_pad) / letter_box_.scale;
  x1 = (x1 - letter_box_.x_pad) / letter_box_.scale;
  y1 = (y1 - letter_box_.y_pad) / letter_box_.scale;

  // Clamp
  x0 = std::max(0.0f, std::min(x0, (float) frame_width_));
  y0 = std::max(0.0f, std::min(y0, (float) frame_height_));
  x1 = std::max(0.0f, std::min(x1, (float) frame_width_));
  y1 = std::max(0.0f, std::min(y1, (float) frame_height_));

  int X = (int) std::round(x0);
  int Y = (int) std::round(y0);
  int W = (int) std::round(std::max(0.0f, x1 - x0));
  int H = (int) std::round(std::max(0.0f, y1 - y0));

  return cv::Rect(X, Y, W, H);
}

std::vector<FaceLocation> FaceDetector::Detect(rga_buffer_t* input_buffer) {
  if (!initialized_) {
    LOGE(TAG, "FaceDetector not initialized");
    return {};
  }
  // auto begin = std::chrono::high_resolution_clock::now();

  // Pre-process Scale
  CalculateScale(input_buffer);

  rga_buffer_t* scaled_buffer = Scale(input_buffer);
  if (scaled_buffer == nullptr) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
    return {};
  }

  // Copy input data to input tensor memory
  rknn_tensor_mem* mem = input_mems_[0];
  if (!mem) {
    LOGE(TAG, "input mem is null");
    return {};
  }

#if USE_QUANTIZED_MODEL
  if (input_tensor_rga_buffer_.handle <= 0) {  // Use DMA buffer, this should not happen
    memcpy(mem->virt_addr, scaled_buffer->vir_addr, mem->size);
  }
#else
  // Convert RGB24 to NHWC float32
  util::rknn::rgb24_to_nhwc_float_stride((uint8_t*) scaled_buffer->vir_addr, model_input_width_,
                                         model_input_height_, (int) scaled_buffer->wstride,
                                         (float*) mem->virt_addr, 1.0f);
#endif

  // auto end = std::chrono::high_resolution_clock::now();
  // auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  // LOGD(TAG, "Pre-process time: %lld ms", preprocess_time);

  // Inference
  auto ret = rknn_run(rknn_ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "rknn_run fail ret=%d", ret);
    return {};
  }

#if ENABLE_PROFILE
  // Print performance info
  rknn_perf_detail perf_info;
  ret = rknn_query(rknn_ctx_, RKNN_QUERY_PERF_DETAIL, &perf_info, sizeof(rknn_perf_detail));
  if (ret == RKNN_SUCC) {
    util::rknn::dump_perf_info(&perf_info);
  }
  rknn_perf_run perf_run_duration;
  ret = rknn_query(rknn_ctx_, RKNN_QUERY_PERF_RUN, &perf_run_duration, sizeof(rknn_perf_run));
  if (ret == RKNN_SUCC) {
    LOGI(TAG, "RKNN Inference Time: %ld us", perf_run_duration.run_duration);
    infer_count++;
    infer_npu_time += perf_run_duration.run_duration;
    LOGD(TAG, "infer count: %ld, cost time: %ld us, average time: %ld us", infer_count,
         infer_npu_time, infer_npu_time / infer_count);
  }
#endif

  // Post-process
  //// Reshape op run on CPU, slower
  // return PostProcess();

  // Prune Transpose & Reshape ops, add extra Sigmoid for score tensors in CPU
  return PostProcessV2();
}

#if USE_QUANTIZED_MODEL

void FaceDetector::DecodeOutput(const int8_t* score_pred, const int8_t* box_pred,
                                const int8_t* lmk_pred, int stride, const std::vector<QntParam>& p,
                                std::vector<FaceLocation>& results) {
  std::vector<float> anchors_center;
  generate_anchors(stride, model_input_width_, 2, anchors_center);

  const QntParam& score_p = p[0];
  const QntParam& box_p = p[1];
  const QntParam& lmk_p = p[2];

  // Sigmoid in NPU model is fused, so we can directly compare with quantized threshold
  int8_t score_thres_i8 =
      util::rknn::qnt_f32_to_affine(det_threshold_, score_p.zero_point, score_p.scale);

  for (int i = 0; i < anchors_center.size() / 2; ++i) {
    auto score = score_pred[i];
    if (score > score_thres_i8) {
      FaceLocation face_info;

      int32_t box_zp = box_p.zero_point;
      float box_scale = box_p.scale;
      // float cx = deqnt_affine_to_f32(anchors_center[i * 2 + 0], box_zp, box_scale);
      // float cy = deqnt_affine_to_f32(anchors_center[i * 2 + 1], box_zp, box_scale);
      float cx = anchors_center[i * 2 + 0];
      float cy = anchors_center[i * 2 + 1];

      float x11 = util::rknn::deqnt_affine_to_f32(box_pred[i * 4 + 0], box_zp, box_scale);
      float x1 = cx - x11 * stride;
      float y11 = util::rknn::deqnt_affine_to_f32(box_pred[i * 4 + 1], box_zp, box_scale);
      float y1 = cy - y11 * stride;
      float x22 = util::rknn::deqnt_affine_to_f32(box_pred[i * 4 + 2], box_zp, box_scale);
      float x2 = cx + x22 * stride;
      float y22 = util::rknn::deqnt_affine_to_f32(box_pred[i * 4 + 3], box_zp, box_scale);
      float y2 = cy + y22 * stride;

      face_info.box = UnScale(x1, y1, x2, y2);  // restore to original coordinate

      face_info.score = score;

      int32_t lmk_zp = lmk_p.zero_point;
      float lmk_scale = lmk_p.scale;
      for (int j = 0; j < 5; ++j) {
        float pxx =
            util::rknn::deqnt_affine_to_f32(lmk_pred[i * 10 + j * 2 + 0], lmk_zp, lmk_scale);
        float px = cx + pxx * stride;
        float pyy =
            util::rknn::deqnt_affine_to_f32(lmk_pred[i * 10 + j * 2 + 1], lmk_zp, lmk_scale);
        float py = cy + pyy * stride;
        face_info.lmk[j * 2 + 0] = px / letter_box_.scale;  // restore scale factor
        face_info.lmk[j * 2 + 1] = py / letter_box_.scale;  // restore scale factor
      }
      results.emplace_back(std::move(face_info));
    }
  }
}

void FaceDetector::DecodeOutputV2(const int8_t* scores_raw, const int8_t* boxes_raw,
                                  const int8_t* lmk_raw, int stride, const std::vector<QntParam>& p,
                                  std::vector<FaceLocation>& results) {
  // Channel-first layout: tensors are [1, C, H, W].
  // For scores: C = num_anchors
  // For boxes: C = num_anchors * 4 (channels ordered by anchor then coord)
  // For lmks: C = num_anchors * 10 (channels ordered by anchor then landmark coord)
  const int num_anchors = 2;

  const QntParam& score_p = p[0];
  const QntParam& box_p = p[1];
  const QntParam& lmk_p = p[2];

  const int H = (model_input_height_ + stride - 1) / stride;
  const int W = (model_input_width_ + stride - 1) / stride;
  const int loc_count = H * W;

  // Determine anchor level index to reuse cached anchors
  int level = 0;
  if (stride == kAnchorStrides[1])
    level = 1;
  else if (stride == kAnchorStrides[2])
    level = 2;
  const std::vector<float>& anchors = anchors_cache_[level];

  // Reserve capacity to avoid repeated reallocations
  results.reserve(results.size() + loc_count * num_anchors / 8);

  const float score_scale = score_p.scale;
  const int32_t score_zp = score_p.zero_point;
  const float box_scale = box_p.scale;
  const int32_t box_zp = box_p.zero_point;
  const float lmk_scale = lmk_p.scale;
  const int32_t lmk_zp = lmk_p.zero_point;

  for (int a = 0; a < num_anchors; ++a) {
    // Channel-first base pointers for this anchor
    const int8_t* score_ch = scores_raw + a * loc_count;
    const int8_t* box_ch_base = boxes_raw + (a * 4) * loc_count;
    const int8_t* lmk_ch_base = lmk_raw + (a * 10) * loc_count;

    for (int s = 0; s < loc_count; ++s) {
      // Dequantize score (logit) and apply sigmoid
      float score_logit = ((float) score_ch[s] - (float) score_zp) * score_scale;
      float score_prob = util::rknn::sigmoid(score_logit);
      if (score_prob <= det_threshold_) continue;

      FaceLocation face_info;

      const int anchor_idx = s * num_anchors + a;
      float cx = anchors[anchor_idx * 2 + 0];
      float cy = anchors[anchor_idx * 2 + 1];

      float x11 = ((float) box_ch_base[0 * loc_count + s] - (float) box_zp) * box_scale;
      float y11 = ((float) box_ch_base[1 * loc_count + s] - (float) box_zp) * box_scale;
      float x22 = ((float) box_ch_base[2 * loc_count + s] - (float) box_zp) * box_scale;
      float y22 = ((float) box_ch_base[3 * loc_count + s] - (float) box_zp) * box_scale;

      float x1 = cx - x11 * stride;
      float y1 = cy - y11 * stride;
      float x2 = cx + x22 * stride;
      float y2 = cy + y22 * stride;

      face_info.box = UnScale(x1, y1, x2, y2);
      face_info.score = score_prob;
      face_info.embedding = {0.f};

      for (int j = 0; j < 5; ++j) {
        float pxx = ((float) lmk_ch_base[(j * 2 + 0) * loc_count + s] - (float) lmk_zp) * lmk_scale;
        float pyy = ((float) lmk_ch_base[(j * 2 + 1) * loc_count + s] - (float) lmk_zp) * lmk_scale;
        float px = cx + pxx * stride;
        float py = cy + pyy * stride;
        face_info.lmk[j * 2 + 0] = px / letter_box_.scale;
        face_info.lmk[j * 2 + 1] = py / letter_box_.scale;
      }

      results.emplace_back(std::move(face_info));
    }
  }
}

std::vector<FaceLocation> FaceDetector::PostProcess() {
  // Output shape:
  // Score:
  // [12800, 1] qnt_type=AFFINE, zp=-128, scale=0.001961
  // [3200, 1] qnt_type=AFFINE, zp=-128, scale=0.001961
  // [800, 1] qnt_type=AFFINE, zp=-128, scale=0.002928
  // Box:
  // [12800, 4] qnt_type=AFFINE, zp=-128, scale=0.013072
  // [3200, 4] qnt_type=AFFINE, zp=-128, scale=0.031143
  // [800, 4] qnt_type=AFFINE, zp=-128, scale=0.026040
  // 5-pair key points(land marks):
  // [12800, 10] qnt_type=AFFINE, zp=-7, scale=0.014990
  // [3200, 10] qnt_type=AFFINE, zp=-12, scale=0.032877
  // [800, 10] qnt_type=AFFINE, zp=-26, scale=0.030569
  std::vector<int8_t*> outputs;
  std::vector<float*> outputs_deqnt;
  outputs.reserve(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    outputs.push_back((int8_t*) output_mems_[i]->virt_addr);
  }

  std::vector<FaceLocation> results;
  for (int i = 0; i < 3; i++) {
    const int8_t* scores = outputs[i];
    const int8_t* boxes = outputs[i + 3];
    const int8_t* lmks = outputs[i + 6];

    DecodeOutput(scores, boxes, lmks, kAnchorStrides[i], deqnt_params_[i], results);
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (const auto& result : results) {
    boxes.push_back(result.box);
    scores.push_back(result.score);
  }

  if (boxes.empty()) {
    // LOGW(TAG, "No valid boxes found after thresholding");
    return {};  // No valid boxes to process
  }

  // Apply Non-Maximum Suppression (NMS)
  std::vector<int> indices;
  indices.reserve(kMaxValidBBoxes);
  cv::dnn::NMSBoxes(boxes, scores, det_threshold_, nms_threshold_, indices, 1.f,
                    kMaxValidBBoxes * 2);

  std::vector<FaceLocation> detected_objects;
  detected_objects.reserve(indices.size());
  for (int idx : indices) {
    detected_objects.emplace_back(results[idx]);
  }

  return detected_objects;
}

std::vector<FaceLocation> FaceDetector::PostProcessV2() {
  // auto begin = std::chrono::high_resolution_clock::now();
  // Output shape:
  // Score:
  // [1, 2, 80, 80] qnt_type=AFFINE, zp=127, scale=0.028093
  // [1, 2, 40, 40] qnt_type=AFFINE, zp=127, scale=0.029802
  // [1, 2, 20, 20] qnt_type=AFFINE, zp=89, scale=0.028445
  // Box:
  // [1, 8, 80, 80] qnt_type=AFFINE, zp=-128, scale=0.013072
  // [1, 8, 40, 40] qnt_type=AFFINE, zp=-128, scale=0.031143
  // [1, 8, 20, 20] qnt_type=AFFINE, zp=-128, scale=0.026040
  // 5-pair key points(land marks):
  // [1, 20, 80, 80] qnt_type=AFFINE, zp=-7, scale=0.014990
  // [1, 20, 40, 40] qnt_type=AFFINE, zp=-12, scale=0.032877
  // [1, 20, 20, 20] qnt_type=AFFINE, zp=-26, scale=0.030569
  std::vector<int8_t*> outputs;
  std::vector<float*> outputs_deqnt;
  outputs.reserve(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    outputs.push_back((int8_t*) output_mems_[i]->virt_addr);
  }

  std::vector<FaceLocation> results;
  for (int i = 0; i < 3; i++) {
    const int8_t* scores_raw = outputs[i];
    const int8_t* boxes_raw = outputs[i + 3];
    const int8_t* lmks_raw = outputs[i + 6];

    DecodeOutputV2(scores_raw, boxes_raw, lmks_raw, kAnchorStrides[i], deqnt_params_[i], results);
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (const auto& result : results) {
    boxes.push_back(result.box);
    scores.push_back(result.score);
  }

  if (boxes.empty()) {
    // LOGW(TAG, "No valid boxes found after thresholding");
    return {};  // No valid boxes to process
  }

  // auto end = std::chrono::high_resolution_clock::now();
  // auto decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  // LOGD(TAG, "Decode time: %lld ms", decode_time);

  // Apply Non-Maximum Suppression (NMS)
  std::vector<int> indices;
  indices.reserve(kMaxValidBBoxes);
  cv::dnn::NMSBoxes(boxes, scores, det_threshold_, nms_threshold_, indices, 1.f,
                    kMaxValidBBoxes * 2);

  std::vector<FaceLocation> detected_objects;
  detected_objects.reserve(indices.size());
  for (int idx : indices) {
    detected_objects.emplace_back(results[idx]);
  }

  // end = std::chrono::high_resolution_clock::now();
  // auto postprocess_time =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  // LOGD(TAG, "Post-process time: %lld ms", postprocess_time);

  return detected_objects;
}

#else

void FaceDetector::DecodeOutput(const float* cls_pred, const float* box_pred, const float* lmk_pred,
                                int stride, std::vector<FaceLocation>& results) {
  std::vector<float> anchors_center;
  generate_anchors(stride, model_input_width_, 2, anchors_center);

  for (int i = 0; i < anchors_center.size() / 2; ++i) {
    float score = cls_pred[i];
    if (score > det_threshold_) {
      FaceLocation face_info;
      float cx = anchors_center[i * 2 + 0];
      float cy = anchors_center[i * 2 + 1];
      float x1 = cx - box_pred[i * 4 + 0] * stride;
      float y1 = cy - box_pred[i * 4 + 1] * stride;
      float x2 = cx + box_pred[i * 4 + 2] * stride;
      float y2 = cy + box_pred[i * 4 + 3] * stride;

      face_info.box = UnScale(x1, y1, x2, y2);  // restore to original coordinate

      face_info.score = score;

      for (int j = 0; j < 5; ++j) {
        float px = cx + lmk_pred[i * 10 + j * 2 + 0] * stride;
        float py = cy + lmk_pred[i * 10 + j * 2 + 1] * stride;
        face_info.lmk[j * 2 + 0] = px / letter_box_.scale;  // restore scale factor
        face_info.lmk[j * 2 + 1] = py / letter_box_.scale;  // restore scale factor
      }
      results.emplace_back(std::move(face_info));
    }
  }
}

std::vector<FaceLocation> FaceDetector::PostProcess() {
  // Output shape:
  // Score:
  // [12800, 1]
  // [3200, 1]
  // [800, 1]
  // Box:
  // [12800, 4]
  // [3200, 4]
  // [800, 4]
  // 5-pair key points(land marks):
  // [12800, 10]
  // [3200, 10]
  // [800, 10]
  std::vector<float*> outputs;
  outputs.reserve(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    outputs.push_back((float*) output_mems_[i]->virt_addr);
  }

  std::vector<FaceLocation> results;
  for (int i = 0; i < 3; i++) {
    const float* scores = outputs[i];
    const float* boxes = outputs[i + 3];
    const float* lmks = outputs[i + 6];

    DecodeOutput(scores, boxes, lmks, kAnchorStrides[i], results);
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (const auto& result : results) {
    boxes.push_back(result.box);
    scores.push_back(result.score);
  }

  if (boxes.empty()) {
    // LOGW(TAG, "No valid boxes found after thresholding");
    return {};  // No valid boxes to process
  }

  // Apply Non-Maximum Suppression (NMS)
  std::vector<int> indices;
  indices.reserve(kMaxValidBBoxes);
  cv::dnn::NMSBoxes(boxes, scores, det_threshold_, nms_threshold_, indices, 1.f,
                    kMaxValidBBoxes * 2);

  std::vector<FaceLocation> detected_objects;
  detected_objects.reserve(indices.size());
  for (int idx : indices) {
    detected_objects.emplace_back(results[idx]);
  }

  return detected_objects;
}
#endif

}  // namespace face