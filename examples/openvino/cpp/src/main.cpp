#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "Common.h"
#include "rknn_api.h"

#define MODEL_PATH "/data/local/tmp/lldb-standalone/student_action_recognition.rknn"
#define TEST_FILE_PATH "/data/local/tmp/lldb-standalone/test_680_400_bgr.bin"

#define TAG "App"

// Input tensor size: 1x3x400x680, float32, NCHW, BGR
constexpr int kInputWidth = 680;
constexpr int kInputHeight = 400;
constexpr size_t kInputTensorSize = 1 * kInputHeight * kInputWidth * 3;
constexpr size_t kInputTensorSizeInBytes = kInputTensorSize * sizeof(float);

struct NormalizedBBox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
};

struct Size {
  uint32_t width;
  uint32_t height;
};

struct SizeF {
  float width;
  float height;
};

struct Rect {
  int32_t x;
  int32_t y;
  uint32_t width;
  uint32_t height;
};

struct DetectedAction {
  /** @brief BBox of detection */
  Rect rect;
  /** @brief Action label */
  int label;
  /** @brief Confidence of detection */
  float detection_conf;
  /** @brief Confidence of predicted action */
  float action_conf;

  DetectedAction(const Rect& rect, int label, float detection_conf, float action_conf)
      : rect(rect), label(label), detection_conf(detection_conf), action_conf(action_conf) {}
};
using DetectedActions = std::vector<DetectedAction>;

constexpr int NUM_ANCHORS = 4;
constexpr SizeF ANCHOR_SIZE[NUM_ANCHORS] = {
    {29.9271, 67.037}, {41.4375, 90.3704}, {58.0833, 129.259}, {91.0208, 190}};
constexpr Size ANCHOR_BLOB_SIZE = {43, 25};
constexpr NormalizedBBox VARIANCES = {0.1f, 0.1f, 0.2f, 0.2f};
constexpr Size FRAME_SIZE = {1920, 1080};
constexpr int NUM_ACTIONS = 3;
constexpr float DETECTION_CONF_THRESHOLD = 0.3f;
constexpr float ACTION_CONF_THRESHOLD = 0.75f;
constexpr size_t TOP_K = 200;
constexpr float NMS_SIGMA = 0.6f;
constexpr int NUM_CANDIDATES = 4300;  // 43*25*4

static NormalizedBBox generate_prior_box(int pos, int step, const SizeF& anchor,
                                         const Size& blob_size) {
  const int row = pos / blob_size.width;
  const int col = pos % blob_size.width;

  const float center_x = (col + 0.5f) * static_cast<float>(step);
  const float center_y = (row + 0.5f) * static_cast<float>(step);

  NormalizedBBox bbox;
  bbox.xmin = (center_x - 0.5f * anchor.width) / (float) kInputWidth;
  bbox.ymin = (center_y - 0.5f * anchor.height) / (float) kInputHeight;
  bbox.xmax = (center_x + 0.5f * anchor.width) / (float) kInputWidth;
  bbox.ymax = (center_y + 0.5f * anchor.height) / (float) kInputHeight;

  return bbox;
}

static NormalizedBBox parse_bbox_record(const float* data) {
  NormalizedBBox bbox;
  bbox.xmin = data[0];
  bbox.ymin = data[1];
  bbox.xmax = data[2];
  bbox.ymax = data[3];
  return bbox;
}

static Rect convert_to_rect(const NormalizedBBox& prior_bbox, const NormalizedBBox& variances,
                            const NormalizedBBox& encoded_bbox, const Size& frame_size) {
  // Convert prior bbox to CV_Rect
  const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
  const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
  const float prior_center_x = 0.5f * (prior_bbox.xmin + prior_bbox.xmax);
  const float prior_center_y = 0.5f * (prior_bbox.ymin + prior_bbox.ymax);

  // Decode bbox coordinates from the SSD format
  const float decoded_bbox_center_x =
      variances.xmin * encoded_bbox.xmin * prior_width + prior_center_x;
  const float decoded_bbox_center_y =
      variances.ymin * encoded_bbox.ymin * prior_height + prior_center_y;
  const float decoded_bbox_width =
      static_cast<float>(exp(static_cast<float>(variances.xmax * encoded_bbox.xmax))) * prior_width;
  const float decoded_bbox_height =
      static_cast<float>(exp(static_cast<float>(variances.ymax * encoded_bbox.ymax))) *
      prior_height;

  // Create decoded bbox
  const float decoded_bbox_xmin = decoded_bbox_center_x - 0.5f * decoded_bbox_width;
  const float decoded_bbox_ymin = decoded_bbox_center_y - 0.5f * decoded_bbox_height;
  const float decoded_bbox_xmax = decoded_bbox_center_x + 0.5f * decoded_bbox_width;
  const float decoded_bbox_ymax = decoded_bbox_center_y + 0.5f * decoded_bbox_height;

  // Convert decoded bbox to CV_Rect
  return Rect{static_cast<int32_t>(decoded_bbox_xmin * frame_size.width),
              static_cast<int32_t>(decoded_bbox_ymin * frame_size.height),
              static_cast<uint32_t>((decoded_bbox_xmax - decoded_bbox_xmin) * frame_size.width),
              static_cast<uint32_t>((decoded_bbox_ymax - decoded_bbox_ymin) * frame_size.height)};
}

static Rect intersection(const Rect& r1, const Rect& r2) {
  Rect inter{0, 0, 0, 0};

  inter.x = std::max(r1.x, r2.x);
  inter.y = std::max(r1.y, r2.y);
  inter.width = std::min(r1.x + r1.width, r2.x + r2.width) - inter.x;
  inter.height = std::min(r1.y + r1.height, r2.y + r2.height) - inter.y;

  // Clamp to zero if no overlap
  if (inter.width < 0) inter.width = 0;
  if (inter.height < 0) inter.height = 0;

  return inter;
}

static void soft_non_max_suppression(const DetectedActions& detections, const float sigma,
                                     size_t top_k, const float min_det_conf,
                                     std::vector<int>* out_indices) {
  std::vector<float> scores(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    scores[i] = detections[i].detection_conf;
  }

  top_k = std::min(top_k, scores.size());

  std::vector<size_t> score_idx(scores.size());
  std::iota(score_idx.begin(), score_idx.end(), 0);
  if (top_k < scores.size()) {
    std::nth_element(score_idx.begin(), score_idx.begin() + top_k, score_idx.end(),
                     [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });
    score_idx.resize(top_k);
  }

  std::vector<float> top_scores(top_k);
  for (size_t i = 0; i < top_k; ++i) {
    top_scores[i] = scores[score_idx[i]];
  }

  out_indices->clear();
  for (size_t iter = 0; iter < top_scores.size(); ++iter) {
    auto best_score_itr = std::max_element(top_scores.begin(), top_scores.end());
    size_t best_idx = std::distance(top_scores.begin(), best_score_itr);
    if (top_scores[best_idx] < min_det_conf) {
      break;
    }
    int anchor_idx = score_idx[best_idx];
    out_indices->emplace_back(anchor_idx);

    // Set score to zero, do NOT erase
    top_scores[best_idx] = 0.0f;

    const auto& rect1 = detections[anchor_idx].rect;
    for (size_t i = 0; i < top_scores.size(); ++i) {
      if (top_scores[i] < min_det_conf) {
        continue;
      }
      const auto& rect2 = detections[score_idx[i]].rect;
      const auto inter = intersection(rect1, rect2);
      float overlap = 0.f;
      if (inter.width > 0 && inter.height > 0) {
        int intersection_area = inter.width * inter.height;
        int rect1_area = rect1.width * rect1.height;
        int rect2_area = rect2.width * rect2.height;
        overlap = static_cast<float>(intersection_area) /
                  static_cast<float>(rect1_area + rect2_area - intersection_area);
      }
      top_scores[i] *= std::exp(-overlap * overlap / sigma);
    }
  }
}

static void dump_tensor_attr(rknn_tensor_attr* attr) {
  LOGD(TAG,
       "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, "
       "type=%s, qnt_type=%s, "
       "zp=%d, scale=%f",
       attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2],
       attr->dims[3], attr->n_elems, attr->size, get_format_string(attr->fmt),
       get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_model(const char* filename, uint32_t* model_size) {
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

static void bgr24_to_nchw_u8(const uint8_t* src, int W, int H, uint8_t* dst) {
  const size_t plane = static_cast<size_t>(W) * H;
  uint8_t* dB = dst + 0 * plane;
  uint8_t* dG = dst + 1 * plane;
  uint8_t* dR = dst + 2 * plane;

  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * W * 3;
    for (int x = 0; x < W; ++x) {
      const uint8_t B = row[3 * x + 0];
      const uint8_t G = row[3 * x + 1];
      const uint8_t R = row[3 * x + 2];
      const size_t idx = static_cast<size_t>(y) * W + x;
      dB[idx] = B;
      dG[idx] = G;
      dR[idx] = R;
    }
  }
}

void bgr24_to_nchw_float(
    const uint8_t* bgr, int W, int H,
    float* out,  // size = 3*W*H
    bool channels_are_bgr = true, float scale = 1.0f / 255.0f,
    float mean[3] = nullptr,  // e.g. {0.0f,0.0f,0.0f} or {0.485f,0.456f,0.406f}
    float stdv[3] = nullptr)  // e.g. {1.0f,1.0f,1.0f} or {0.229f,0.224f,0.225f}
{
  const size_t plane = (size_t) W * H;
  float* d0 = out + 0 * plane;
  float* d1 = out + 1 * plane;
  float* d2 = out + 2 * plane;

  const float m[3] = {mean ? mean[0] : 0.f, mean ? mean[1] : 0.f, mean ? mean[2] : 0.f};
  const float s[3] = {stdv ? stdv[0] : 1.f, stdv ? stdv[1] : 1.f, stdv ? stdv[2] : 1.f};

  for (int y = 0; y < H; ++y) {
    const uint8_t* row = bgr + (size_t) y * W * 3;
    for (int x = 0; x < W; ++x) {
      uint8_t B = row[3 * x + 0];
      uint8_t G = row[3 * x + 1];
      uint8_t R = row[3 * x + 2];
      const size_t idx = (size_t) y * W + x;

      // If model expects BGR order in NCHW (common with OpenCV-trained models)
      if (channels_are_bgr) {
        d0[idx] = (B * scale - m[0]) / s[0];
        d1[idx] = (G * scale - m[1]) / s[1];
        d2[idx] = (R * scale - m[2]) / s[2];
      } else {  // model expects RGB order
        d0[idx] = (R * scale - m[0]) / s[0];
        d1[idx] = (G * scale - m[1]) / s[1];
        d2[idx] = (B * scale - m[2]) / s[2];
      }
    }
  }
}

struct App {
  std::atomic<bool> running;

  rknn_context rknn_ctx;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  rknn_input_output_num io_num;
  std::vector<rknn_tensor_mem*> input_mems;
  std::vector<rknn_tensor_mem*> output_mems;

  App(const char* model_path);
  ~App();

  int Init(const char* model_path);
  void DeInit();
  int Inference();
  void PostProcess(const std::vector<rknn_output>& outputs);
};

std::unique_ptr<App> app_instance;

App::App(const char* model_path) : running(false) {
  auto ret = Init(model_path);
  if (ret == 0) {
    LOGI(TAG, "App initialized successfully");
  } else {
    assert(false && "App initialization failed");
  }
}

App::~App() {
  DeInit();
}

int App::Init(const char* model_path) {
  int ret = 0;

  // Load model
  uint32_t model_len = 0;
  auto model = load_model(MODEL_PATH, &model_len);
  if (model == nullptr) {
    LOGE(TAG, "load model failed");
    return -1;
  }

  // Init RKNN
  ret = rknn_init(&rknn_ctx, model, model_len, 0, nullptr);
  if (ret < 0) {
    LOGE(TAG, "rknn_init fail ret=%d", ret);

    free(model);
    return -1;
  }

  // Query IO number
  ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(rknn_input_output_num));
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "rknn_query fail ret=%d", ret);
    return -1;
  }
  LOGI(TAG, "rknn_query input num: %d, output num: %d", io_num.n_input, io_num.n_output);

  // Query input tensor shape
  std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;  // Set index for input tensor
    ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_query input attr fail ret=%d", ret);
      return -1;
    }

    dump_tensor_attr(&input_attrs[i]);
  }

  // Query output tensor shape
  std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;  // Set index for output tensor
    ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_query output attr fail ret=%d", ret);
      return -1;
    }

    dump_tensor_attr(&output_attrs[i]);
  }

  // // Create input tensors
  // input_mems.reserve(io_num.n_input);
  // for (int i = 0; i < io_num.n_input; i++) {
  //   input_attrs[i].fmt = RKNN_TENSOR_NCHW;  // Set format to NCHW
  //   input_attrs[i].type = RKNN_TENSOR_FLOAT32; // float32, not quantized

  //   // Create memory for this tensor
  //   auto* mem = rknn_create_mem(rknn_ctx, kInputTensorSizeInBytes);
  //   if (!mem) {
  //     LOGE(TAG, "create input mem fail");
  //     return -1;
  //   }
  //   input_mems.push_back(mem);

  //   // Set input tensor memory with attributes
  //   ret = rknn_set_io_mem(rknn_ctx, mem, &input_attrs[i]);
  //   if (ret != RKNN_SUCC) {
  //     LOGE(TAG, "rknn_set_io_mem fail ret=%d", ret);
  //     return ret;
  //   }
  // }

  // // Create output tensors
  // output_mems.reserve(io_num.n_output);
  // for (int i = 0; i < io_num.n_output; i++) {
  //   output_attrs[i].type = RKNN_TENSOR_FLOAT32; // float32, not quantized

  //   // Create memory for this tensor
  //   auto* mem = rknn_create_mem(rknn_ctx, output_attrs[i].size);
  //   if (!mem) {
  //     LOGE(TAG, "create output mem fail");
  //     return -1;
  //   }
  //   output_mems.push_back(mem);

  //   // Set output tensor memory with attributes
  //   ret = rknn_set_io_mem(rknn_ctx, mem, &output_attrs[i]);
  //   if (ret != RKNN_SUCC) {
  //     LOGE(TAG, "rknn_set_io_mem fail ret=%d", ret);
  //     return ret;
  //   }
  // }

  return 0;
}

void App::DeInit() {
  // Free input and output memory
  for (auto* mem : input_mems) {
    if (mem) {
      rknn_destroy_mem(rknn_ctx, mem);
    }
  }
  for (auto* mem : output_mems) {
    if (mem) {
      rknn_destroy_mem(rknn_ctx, mem);
    }
  }

  // Deinitialize RKNN context
  rknn_destroy(rknn_ctx);

  LOGI(TAG, "App deinitialized successfully");
}

int App::Inference() {
  // Read entire input
  std::ifstream fin(TEST_FILE_PATH, std::ios::binary | std::ios::ate);
  if (!fin) {
    LOGE(TAG, "Failed to open input: %s", TEST_FILE_PATH);
    return -1;
  }
  const std::streamsize fsz = fin.tellg();
  fin.seekg(0, std::ios::beg);

  if (fsz != static_cast<std::streamsize>(kInputTensorSize)) {
    LOGE(TAG, "Input file size %ld does not match expected size %zu", fsz, kInputTensorSize);
    return -2;
  }

  std::vector<uint8_t> bgr(kInputTensorSize);
  if (!fin.read(reinterpret_cast<char*>(bgr.data()), fsz)) {
    LOGE(TAG, "Failed to read input data");
    return -3;
  }
  fin.close();

  // Convert to NCHW uint8
  const size_t plane = static_cast<size_t>(kInputWidth) * kInputHeight;
  std::vector<uint8_t> nchw_u8(3 * plane);
  bgr24_to_nchw_u8(bgr.data(), kInputWidth, kInputHeight, nchw_u8.data());

  // Convert to float32 and normalize
  std::vector<float> nchw_f32(kInputTensorSize);
  bgr24_to_nchw_float(nchw_u8.data(), kInputWidth, kInputHeight, nchw_f32.data());

  // Prepare input
  std::vector<rknn_input> inputs(io_num.n_input);
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_FLOAT32;
  inputs[0].fmt = RKNN_TENSOR_NCHW;
  inputs[0].size = kInputTensorSizeInBytes;
  inputs[0].buf = nchw_f32.data();

  auto ret = rknn_inputs_set(rknn_ctx, io_num.n_input, inputs.data());
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "failed to set input, ret=%d", ret);
    return false;
  }

  // Run inference
  ret = rknn_run(rknn_ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "rknn_run fail ret=%d", ret);
    return ret;
  }

  LOGI(TAG, "Inference completed successfully");

  // Get output
  std::vector<rknn_output> outputs(io_num.n_output);
  for (int i = 0; i < io_num.n_output; i++) {
    outputs[i].index = i;
    outputs[i].want_float = true;
  }
  ret = rknn_outputs_get(rknn_ctx, io_num.n_output, outputs.data(), nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "failed to get output, ret=%d", ret);
    return false;
  }

  // Post-process output
  PostProcess(outputs);

  return 0;
}

// index=0, name=bboxes, n_dims=2, dims=[1, 17200], n_elems=17200, size=34400
// index=1, name=bboxes_scores, n_dims=2, dims=[1, 8600], n_elems=8600, size=17200
// index=2, name=anchor3, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
// index=3, name=anchor2, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
// index=4, name=anchor1, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
// index=5, name=anchor4, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
void App::PostProcess(const std::vector<rknn_output>& outputs) {
  // Bounding boxes
  const float* bboxes = reinterpret_cast<const float*>(outputs[0].buf);

  // Bounding box scores
  const float* bbox_scores = reinterpret_cast<const float*>(outputs[1].buf);

  // Anchors, order: [4,2,3,5]
  const float* anchors[NUM_ANCHORS] = {
      reinterpret_cast<const float*>(outputs[4].buf),
      reinterpret_cast<const float*>(outputs[2].buf),
      reinterpret_cast<const float*>(outputs[3].buf),
      reinterpret_cast<const float*>(outputs[5].buf),
  };

  DetectedActions valid_detections;
  for (int p = 0; p < NUM_CANDIDATES; ++p) {
    float detection_conf = bbox_scores[p * 2 + 1];  // score for person detection
    if (detection_conf < DETECTION_CONF_THRESHOLD) {
      continue;
    }

    int anchor_id = p % NUM_ANCHORS;
    const float* anchor_ptr = anchors[anchor_id];
    auto anchor_idx = p / NUM_ANCHORS * NUM_ACTIONS;

    int32_t action_label = -1;
    float action_max_exp_value = 0.f;
    float action_sum_exp_values = 0.f;

    for (int32_t c = 0; c < NUM_ACTIONS; c++) {
      float exp_value = std::exp(anchor_ptr[anchor_idx + c] * 3.f);
      action_sum_exp_values += exp_value;
      if (exp_value > action_max_exp_value) {
        action_max_exp_value = exp_value;
        action_label = c;
      }
    }

    if (std::fabs(action_sum_exp_values) < std::numeric_limits<float>::epsilon()) {
      LOGE(TAG, "action_sum_exp_values can't be equal to 0");
      break;
    }

    // Estimate the action confidence
    float action_conf = action_max_exp_value / action_sum_exp_values;

    // Skip low-confidence actions
    if (action_label < 0 || action_conf < ACTION_CONF_THRESHOLD) {
      action_label = 0;
      action_conf = 0.f;
    }

    // Prior boxes
    const auto prior_boxes =
        generate_prior_box(p / NUM_ANCHORS, 16, ANCHOR_SIZE[anchor_id], ANCHOR_BLOB_SIZE);
    // Encoded boxes
    const auto encoded_boxes = parse_bbox_record(bboxes + p * NUM_ANCHORS);

    const auto det_rect = convert_to_rect(prior_boxes, VARIANCES, encoded_boxes, FRAME_SIZE);
    // Store detected action
    valid_detections.emplace_back(det_rect, action_label, detection_conf, action_conf);
  }

  LOGD(TAG, "Detections before NMS: %lu", valid_detections.size());

  // Merge most overlapped detections
  std::vector<int> out_det_indices;
  soft_non_max_suppression(valid_detections, NMS_SIGMA, TOP_K, DETECTION_CONF_THRESHOLD,
                           &out_det_indices);

  LOGI(TAG, "Detections after NMS: %lu", out_det_indices.size());
}

// handle abnormal termination signals
static void handle_sig(int32_t signo) {
  if (signo == SIGINT || signo == SIGTERM) {
    // handle graceful shutdown
    LOGI(TAG, "Received signal %d, shutting down...", signo);
    if (app_instance) {
      app_instance->running.store(false);
    }
  }
}

int main(int argc, char* argv[]) {
  // Check files are accessible
  if (!std::filesystem::exists(MODEL_PATH)) {
    LOGE(TAG, "Model file not found: %s", MODEL_PATH);
    return -1;
  }
  if (!std::filesystem::exists(TEST_FILE_PATH)) {
    LOGE(TAG, "Test file not found: %s", TEST_FILE_PATH);
    return -1;
  }

  // Add SIG handlers for graceful shutdown.
  struct sigaction sa = {};
  sa.sa_handler = handle_sig;
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  // Create Application
  app_instance = std::make_unique<App>(MODEL_PATH);

  // Enter main loop
  app_instance->running.store(true);
  while (app_instance->running.load()) {
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    app_instance->Inference();
  }

  // Release resources
  app_instance.reset();

  return 0;
}