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
#include <unordered_map>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Common.h"
#include "rknn_api.h"
#include "utils/MppDecoder.h"
#include "utils/MppEncoder.h"
#include "utils/RgaBufferPool.h"

#define MODEL_PATH "/data/local/tmp/lldb-standalone/student_action_recognition.rknn"
#define TEST_BIN_FILE_PATH "/data/local/tmp/lldb-standalone/test_680_400_bgr.bin"
#define TEST_IMG_FILE_PATH "/data/local/tmp/lldb-standalone/test_680_400.jpg"
#define TEST_VID_FILE_PATH "/data/local/tmp/lldb-standalone/test_1920_1080.h264"
#define BGR_FILE_PATH "/data/local/tmp/lldb-standalone/inference_680_400_bgr.bmp"

#define TAG "App"

#define TEST_WITH_IMAGE 0
#define USE_ZEROS_COPY 1

struct NormalizedBBox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
};

struct DetectedAction {
  cv::Rect rect;
  int label;
  float detection_conf;
  float action_conf;

  DetectedAction(const cv::Rect& rect, int label, float detection_conf, float action_conf)
      : rect(rect), label(label), detection_conf(detection_conf), action_conf(action_conf) {}
  ~DetectedAction() = default;
};

using DetectedActions = std::vector<DetectedAction>;

// Input tensor size: 1x3x400x680, float32, NCHW, BGR
constexpr int kInputWidth = 680;
constexpr int kInputHeight = 400;
constexpr size_t kInputTensorSize = 1 * kInputHeight * kInputWidth * 3;
constexpr size_t kInputTensorSizeInBytes = kInputTensorSize * sizeof(float);
constexpr int NUM_ANCHORS = 4;
constexpr NormalizedBBox VARIANCES = {0.1f, 0.1f, 0.2f, 0.2f};
static cv::Size2f ANCHOR_SIZE[NUM_ANCHORS] = {
    {29.9271, 67.037}, {41.4375, 90.3704}, {58.0833, 129.259}, {91.0208, 190}};
static cv::Size FRAME_SIZE = {1920, 1080};
static cv::Size2f ANCHOR_BLOB_SIZE = {43, 25};
constexpr int NUM_ACTIONS = 3;
constexpr float DETECTION_CONF_THRESHOLD = 0.3f;
constexpr float ACTION_CONF_THRESHOLD = 0.75f;
constexpr size_t TOP_K = 200;
constexpr float NMS_SIGMA = 0.6f;
constexpr int NUM_CANDIDATES = 4300;  // 43*25*4
constexpr const char* ACTION_CLASS_LIST[NUM_ACTIONS] = {"Sitting", "Standing", "RaisingHand"};

static NormalizedBBox generate_prior_box(int pos, int step, const cv::Size2f& anchor,
                                         const cv::Size& blob_size) {
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

static cv::Rect convert_to_rect(const NormalizedBBox& prior_bbox, const NormalizedBBox& variances,
                                const NormalizedBBox& encoded_bbox, const cv::Size& frame_size) {
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
  return cv::Rect{
      static_cast<int32_t>(decoded_bbox_xmin * frame_size.width),
      static_cast<int32_t>(decoded_bbox_ymin * frame_size.height),
      static_cast<int32_t>((decoded_bbox_xmax - decoded_bbox_xmin) * frame_size.width),
      static_cast<int32_t>((decoded_bbox_ymax - decoded_bbox_ymin) * frame_size.height)};
}

static void soft_non_max_suppression(const DetectedActions& detections, const float sigma,
                                     size_t top_k, const float min_det_conf,
                                     std::vector<int>* out_indices) {
  // Store input bbox scores
  std::vector<float> scores(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    scores[i] = detections[i].detection_conf;
  }

  top_k = std::min(top_k, scores.size());

  // Select top-k score indices
  std::vector<size_t> score_idx(scores.size());
  std::iota(score_idx.begin(), score_idx.end(), 0);
  std::nth_element(score_idx.begin(), score_idx.begin() + top_k, score_idx.end(),
                   [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

  // Extract top-k score values
  std::vector<float> top_scores(top_k);
  for (size_t i = 0; i < top_scores.size(); ++i) {
    top_scores[i] = scores[score_idx[i]];
  }

  // Carry out Soft Non-Maximum Suppression algorithm
  out_indices->clear();
  for (size_t step = 0; step < top_scores.size(); ++step) {
    auto best_score_itr = std::max_element(top_scores.begin(), top_scores.end());
    if (*best_score_itr < min_det_conf) {
      break;
    }

    // Add current bbox to output list
    const size_t local_anchor_idx = std::distance(top_scores.begin(), best_score_itr);
    const int anchor_idx = score_idx[local_anchor_idx];
    out_indices->emplace_back(anchor_idx);
    *best_score_itr = 0.f;

    // Update top_scores of the rest bboxes
    for (size_t local_reference_idx = 0; local_reference_idx < top_scores.size();
         ++local_reference_idx) {
      // Skip updating step for the low-confidence bbox
      if (top_scores[local_reference_idx] < min_det_conf) {
        continue;
      }

      // Calculate the Intersection over Union metric between two bboxes
      const size_t reference_idx = score_idx[local_reference_idx];
      const auto& rect1 = detections[anchor_idx].rect;
      const auto& rect2 = detections[reference_idx].rect;
      const auto intersection = rect1 & rect2;
      float overlap = 0.f;
      if (intersection.width > 0 && intersection.height > 0) {
        const int intersection_area = intersection.area();
        overlap = static_cast<float>(intersection_area) /
                  static_cast<float>(rect1.area() + rect2.area() - intersection_area);
      }

      // Scale bbox score using the exponential rule
      top_scores[local_reference_idx] *= std::exp(-overlap * overlap / sigma);
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

static void bgr24_to_nchw_float(const uint8_t* src, int W, int H, float* dst, float scale) {
  const size_t plane = static_cast<size_t>(W) * H;
  float* dB = dst + 0 * plane;
  float* dG = dst + 1 * plane;
  float* dR = dst + 2 * plane;

  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * W * 3;
    for (int x = 0; x < W; ++x) {
      const size_t idx = static_cast<size_t>(y) * W + x;
      dB[idx] = row[3 * x + 0] * scale;  // B
      dG[idx] = row[3 * x + 1] * scale;  // G
      dR[idx] = row[3 * x + 2] * scale;  // R
    }
  }
}

static void bgr24_to_nhwc_float(const uint8_t* src, int W, int H, float* dst, float scale) {
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * W * 3;
    for (int x = 0; x < W; ++x) {
      const size_t idx = (static_cast<size_t>(y) * W + x) * 3;
      dst[idx + 0] = row[3 * x + 0] * scale;  // B
      dst[idx + 1] = row[3 * x + 1] * scale;  // G
      dst[idx + 2] = row[3 * x + 2] * scale;  // R
    }
  }
}

// Convert BGR24 to NHWC float32 with stride support

static void bgr24_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride, float* dst,
                                       float scale) {
  // stride: number of bytes per row (may be larger than W*3)
  // NHWC layout: dst[(y * W + x) * 3 + c]
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * stride * 3;
    for (int x = 0; x < W; ++x) {
      const size_t idx = ((size_t) y * W + x) * 3;
      dst[idx + 0] = static_cast<float>(row[3 * x + 0]) * scale;  // B
      dst[idx + 1] = static_cast<float>(row[3 * x + 1]) * scale;  // G
      dst[idx + 2] = static_cast<float>(row[3 * x + 2]) * scale;  // R
    }
  }
}

// Convert BGR24 to NCHW float32 with stride support
static void bgr24_to_nchw_float_stride(const uint8_t* src, int W, int H, int stride, float* dst,
                                       float scale) {
  // stride: number of bytes per row (may be larger than W*3)
  // NCHW layout: dst[c * H * W + y * W + x]
  size_t plane = (size_t) W * H;
  float* dB = dst + 0 * plane;
  float* dG = dst + 1 * plane;
  float* dR = dst + 2 * plane;
  for (int y = 0; y < H; ++y) {
    const uint8_t* row = src + (size_t) y * stride * 3;
    for (int x = 0; x < W; ++x) {
      size_t idx = (size_t) y * W + x;
      dB[idx] = static_cast<float>(row[3 * x + 0]) * scale;
      dG[idx] = static_cast<float>(row[3 * x + 1]) * scale;
      dR[idx] = static_cast<float>(row[3 * x + 2]) * scale;
    }
  }
}

static bool read_file(const std::string& path, std::vector<uint8_t>& buf) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    LOGE(TAG, "open file %s failed", path.c_str());
    return false;
  }
  std::streamsize sz = f.tellg();
  if (sz <= 0) {
    LOGE(TAG, "file %s is empty", path.c_str());
    return false;
  }
  buf.resize(static_cast<size_t>(sz));
  f.seekg(0, std::ios::beg);
  return (bool) f.read(reinterpret_cast<char*>(buf.data()), sz);
}

static void bgr_to_nchw_float_no_norm(const cv::Mat& img, std::vector<float>& out) {
  CV_Assert(img.type() == CV_8UC3 && img.cols == kInputWidth && img.rows == kInputHeight);
  const int W = kInputWidth, H = kInputHeight;
  const size_t plane = (size_t) W * H;
  out.resize(3 * plane);  // 1CHW

  float* dB = out.data() + 0 * plane;
  float* dG = out.data() + 1 * plane;
  float* dR = out.data() + 2 * plane;

  for (int y = 0; y < H; ++y) {
    const uint8_t* row = img.ptr<uint8_t>(y);
    for (int x = 0; x < W; ++x) {
      const size_t idx = (size_t) y * W + x;
      dB[idx] = (float) row[3 * x + 0];  // 0..255
      dG[idx] = (float) row[3 * x + 1];
      dR[idx] = (float) row[3 * x + 2];
    }
  }
}

void bgr_to_nhwc_float_no_norm(const cv::Mat& img, std::vector<float>& out) {
  // CV_Assert(img.type() == CV_8UC3 && img.cols == kInputWidth && img.rows == kInputHeight);
  const int W = kInputWidth, H = kInputHeight;
  out.resize(1 * H * W * 3);  // NHWC

  float* dst = out.data();

  for (int y = 0; y < H; ++y) {
    const uint8_t* row = img.ptr<uint8_t>(y);
    for (int x = 0; x < W; ++x) {
      // NHWC index: (y, x, c)
      size_t idx = ((size_t) y * W + x) * 3;
      dst[idx + 0] = static_cast<float>(row[3 * x + 0]);  // B
      dst[idx + 1] = static_cast<float>(row[3 * x + 1]);  // G
      dst[idx + 2] = static_cast<float>(row[3 * x + 2]);  // R
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
// App class
struct VideoFileReader;  // Forward declaration
struct VideoFileWriter;  // Forward declaration
struct App {
  std::atomic<bool> running;

  rknn_context rknn_ctx;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  rknn_input_output_num io_num;
  std::vector<rknn_tensor_mem*> input_mems;
  std::vector<rknn_tensor_mem*> output_mems;

  std::vector<float> input_data;
  bool input_data_ready = false;

  std::unique_ptr<VideoFileReader> capturer = nullptr;
  std::unique_ptr<std::thread> worker = nullptr;

  explicit App(const char* model_path);
  ~App();

  int Init(const char* model_path);
  void DeInit();
  int Inference(const rga_buffer_t* buffer);
  void PostProcess(const float* bboxes, const float* scores, const float* const* anchors);
  void OnDecodedFrame(const VideoFrameSlot& frame);

  bool Start();
  void Stop();
  void MainLoop();

#if TEST_WITH_IMAGE
  // Load input data from binary file
  int LoadTestData(const std::string& bin_path, std::vector<float>& input_data);
  // Load input data from image file(backend is OpenCV)
  int LoadTestImage(const std::string& img_path, std::vector<float>& input_data);
#endif

 private:
  // buffers
  std::unique_ptr<RingBuffer<VideoFrameSlot>> ring_buffer_;
  std::unique_ptr<RgaBufferPool> scale_buffer_pool_;
  std::unique_ptr<RgaBufferPool> rgb_buffer_pool_;
  std::unordered_map<int, std::unique_ptr<rga_buffer_t>> rga_buffers_;
  int last_success_fd_;
  std::atomic<bool> ready_;

  static bool ImportRgaBuffer(const VideoFrameSlot& frame, rga_buffer_t* buffer);
  rga_buffer_t* ScaleConvert(rga_buffer_t* src);
  rga_buffer_t* GetImportedRgaBuffer();
};

/////////////////////////////////////////////////////////////////////////////////////////
// VideoFileReader: Capture video frames from a H.264 encoded video file, decode
struct VideoFileReader {
  std::atomic<bool> running;
  std::unique_ptr<std::thread> worker;
  std::string path;
  std::unique_ptr<v_dec::MppDecoder> decoder;
  uint32_t frame_width;
  uint32_t frame_height;
  uint16_t fps;
  bool cycle_mode;

  explicit VideoFileReader(const std::string& path, void* user_data);
  ~VideoFileReader();

  bool Start();
  void Stop();

 private:
  void CaptureLoop();
  static void SplitH264Packet(int32_t* read_len, uint8_t* buf, size_t buf_size, int32_t used_bytes);
};

/////////////////////////////////////////////////////////////////////////////////////////
// VideoFileWriter: Write video frames to a H.264 encoded video file
struct VideoFileWriter {
  std::string path;
  std::unique_ptr<v_enc::MppEncoder> encoder;
  EncodedVideoInfo enc_info;

  explicit VideoFileWriter(const std::string& path, uint32_t width, uint32_t height, int fps);
  ~VideoFileWriter();

  void WriteFrame(const VideoFrameSlot& frame);

 private:
  FILE* fp_;
  void OnEncodedData(const char* data, size_t size);
};

/////////////////////////////////////////////////////////////////////////////////////////

VideoFileWriter::VideoFileWriter(const std::string& path, uint32_t width, uint32_t height, int fps)
    : path(path), enc_info{} {
  fp_ = fopen(path.c_str(), "wb");
  if (fp_ == nullptr) {
    LOGE(TAG, "Failed to open output file: %s", path.c_str());
    assert(false);
  }

  enc_info.width = width;
  enc_info.height = height;
  enc_info.fps = fps;
  enc_info.codec = 0;  // H.264
  encoder = std::make_unique<v_enc::MppEncoder>(enc_info, this);
  encoder->SetCallback([](void* userdata, const char* data, size_t size) {
    auto* writer = reinterpret_cast<VideoFileWriter*>(userdata);
    writer->OnEncodedData(data, size);
  });
}

VideoFileWriter::~VideoFileWriter() {
  if (encoder) {
    encoder->Stop();
    encoder.reset();
  }

  if (fp_) {
    fflush(fp_);
    fclose(fp_);
    fp_ = nullptr;
  }
}

void VideoFileWriter::WriteFrame(const VideoFrameSlot& frame) {
  if (encoder) {
    encoder->Encode(frame);
  }
}

void VideoFileWriter::OnEncodedData(const char* data, size_t size) {
  if (fp_ == nullptr) {
    LOGE(TAG, "Can not write to file: %s", path.c_str());
    return;
  }

  if (data && size > 0) {
    fwrite(data, 1, size, fp_);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////

VideoFileReader::VideoFileReader(const std::string& filepath, void* user_data)
    : running(false),
      worker(nullptr),
      path(filepath),
      decoder(std::make_unique<v_dec::MppDecoder>(MPP_VIDEO_CodingAVC, kBaseVideoWidth,
                                                  kBaseVideoHeight, user_data)),
      frame_width(kBaseVideoWidth),
      frame_height(kBaseVideoHeight),
      fps(25),
      cycle_mode(true) {
  decoder->SetCallback([](void* userdata, RK_U32 width_stride, RK_U32 height_stride, RK_U32 width,
                          RK_U32 height, MppFrameFormat format, int fd) {
    auto* app = reinterpret_cast<App*>(userdata);
    app->OnDecodedFrame({width, height, width_stride, height_stride, format, fd});
  });
}

VideoFileReader::~VideoFileReader() {
  Stop();
}

bool VideoFileReader::Start() {
  if (running.load()) {
    LOGW(TAG, "video capturer already running");
    return false;
  }

  running.store(true);
  worker = std::make_unique<std::thread>(&VideoFileReader::CaptureLoop, this);
  return true;
}

void VideoFileReader::Stop() {
  if (!running.load()) {
    LOGW(TAG, "video capturer not running");
    return;
  }

  // send EOS to video decoder
  if (decoder != nullptr) {
    decoder->PutPacket(nullptr, 0, 1);
  }

  // stop capture thread
  running.store(false);
  if (worker && worker->joinable()) {
    worker->join();
    worker.reset();
  }
}

void VideoFileReader::SplitH264Packet(int32_t* read_len, uint8_t* buf, size_t buf_size,
                                      int32_t used_bytes) {
  int32_t i;
  bool find_start = false;
  bool find_end = false;
  bool new_pic;
  /* H264 frame start marker */
  if (*read_len > buf_size) {
    LOGE(TAG, "Read length %d exceeds minimum buffer size %lu", *read_len, buf_size);
    return;
  }

  for (i = 0; i < *read_len - 8; i++) { /* 8:h264 frame start code length */
    int tmp = buf[i + 3] & 0x1F;        /* 3:index  0x1F:frame start marker */
    new_pic =
        (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && /* 1 2:index */
         (((tmp == 0x5 || tmp == 0x1) &&
           ((buf[i + 4] & 0x80) == 0x80)) ||            /* 4:index 0x5 0x80:frame start mark */
          (tmp == 20 && (buf[i + 7] & 0x80) == 0x80))); /* 20 0x1 0x80:frame start marker 7:index */
    if (new_pic) {
      find_start = true;
      i += 8; /* 8:h264 frame start code length */
      break;
    }
  }

  for (; i < *read_len - 8; i++) { /* 8:h264 frame start code length */
    int tmp = buf[i + 3] & 0x1F;   /* 3:index  0x1F:frame start marker */
    new_pic =
        (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 && /* 1 2:index */
         (tmp == 15 || tmp == 7 || tmp == 8 || tmp == 6 ||    /* 15 7 8 6:frame start marker */
          ((tmp == 5 || tmp == 1) &&
           ((buf[i + 4] & 0x80) == 0x80)) ||            /* 4:index 5 0x80:frame start marker */
          (tmp == 20 && (buf[i + 7] & 0x80) == 0x80))); /* 7:index 20 0x80:frame start marker */
    if (new_pic) {
      find_end = true;
      break;
    }
  }

  if (i > 0) {
    *read_len = i;
  }
  if (!find_start) {
    LOGE(TAG, "Cannot find H264 start code! Read length: %d, Used bytes: %d", *read_len,
         used_bytes);
  }
  if (!find_end) {
    *read_len = i + 8; /* 8:h264 frame start code length */
  }
  return;
}

void VideoFileReader::CaptureLoop() {
  FILE* input_file = fopen(path.c_str(), "rb");
  if (!input_file) {
    LOGE(TAG, "Failed to open input file: %s", path.c_str());
    return;
  }
  LOGI(TAG, "Opened input file: %s", path.c_str());

  const size_t buffer_size = (frame_width * frame_height * 3) >> 1;
  uint8_t* buffer = new uint8_t[buffer_size];
  int32_t used_bytes = 0, read_len = 0;

  LOGD(TAG, "Video capture loop started");
  while (running.load()) {
    auto begin = std::chrono::steady_clock::now();

    // Read H.264 frame from file
    fseek(input_file, used_bytes, SEEK_SET);
    read_len = fread(buffer, 1, buffer_size, input_file);

    if (read_len == 0) {
      if (!cycle_mode) {
        LOGI(TAG, "End of stream reached, exiting capture loop");
        break;
      }

      LOGW(TAG, "End of stream reached, restarting from the beginning of the file: %s",
           path.c_str());

      used_bytes = 0;  // Reset to start reading from the beginning
      fseek(input_file, 0, SEEK_SET);
      read_len = fread(buffer, 1, buffer_size, input_file);
    }

    // Split buffer
    SplitH264Packet(&read_len, buffer, buffer_size, used_bytes);

    // Decode frame
    decoder->PutPacket(buffer, read_len, 0);

    used_bytes += read_len;

    // Sleep to simulate real-time frame rate
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    auto frame_duration_ms = 1000 / fps;
    if (elapsed_ms < frame_duration_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(frame_duration_ms - elapsed_ms));
    } else {
      LOGW(TAG, "Processing time %lld ms exceeds frame duration %d ms", elapsed_ms,
           frame_duration_ms);
    }
  }

  delete[] buffer;
  fclose(input_file);

  LOGD(TAG, "Video capture loop exited");
}

////////////////////////////////////////////////////////////////////////////////
App::App(const char* model_path) : running(false), input_data(kInputTensorSize) {
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

#if USE_ZEROS_COPY
  // Create input tensors
  input_mems.reserve(io_num.n_input);
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].fmt = RKNN_TENSOR_NHWC;      // Set format to NHWC
    input_attrs[i].type = RKNN_TENSOR_FLOAT32;  // float32, not quantized
    input_attrs[i].size = kInputTensorSizeInBytes;
    input_attrs[i].size_with_stride = kInputTensorSizeInBytes;

    // Create memory for this tensor
    auto* mem = rknn_create_mem(rknn_ctx, kInputTensorSizeInBytes);
    if (!mem) {
      LOGE(TAG, "create input mem fail");
      return -1;
    }
    input_mems.push_back(mem);

    // Set input tensor memory with attributes
    ret = rknn_set_io_mem(rknn_ctx, mem, &input_attrs[i]);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_set_io_mem fail ret=%d", ret);
      return ret;
    }
  }

  // Create output tensors
  output_mems.reserve(io_num.n_output);
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].type = RKNN_TENSOR_FLOAT32;  // float32, not quantized

    // Calculate output tensor size
    if (strcmp(output_attrs[i].name, "bboxes") == 0) {
      output_attrs[i].size = output_attrs[i].dims[1] * sizeof(float);
      output_attrs[i].size_with_stride = output_attrs[i].size;
    } else if (strcmp(output_attrs[i].name, "bboxes_scores") == 0) {
      output_attrs[i].size = output_attrs[i].dims[1] * sizeof(float);
      output_attrs[i].size_with_stride = output_attrs[i].size;
    } else {
      output_attrs[i].size = output_attrs[i].dims[1] * output_attrs[i].dims[2] *
                             output_attrs[i].dims[3] * sizeof(float);
      output_attrs[i].size_with_stride = output_attrs[i].size;
    }

    // Create memory for this tensor
    auto* mem = rknn_create_mem(rknn_ctx, output_attrs[i].size);
    if (!mem) {
      LOGE(TAG, "create output mem fail");
      return -1;
    }
    output_mems.push_back(mem);

    // Set output tensor memory with attributes
    ret = rknn_set_io_mem(rknn_ctx, mem, &output_attrs[i]);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_set_io_mem fail ret=%d", ret);
      return ret;
    }
  }
#endif

#if TEST_WITH_IMAGE
  // Load input data from binary file
  if (LoadTestData(TEST_BIN_FILE_PATH, input_data) != 0) {
    LOGE(TAG, "Failed to load input data from binary file");
    return -1;
  }
  input_data_ready = true;
#else
  // create ring buffer
  ring_buffer_ = std::make_unique<RingBuffer<VideoFrameSlot>>(kDefaultRingBufferSize);
  // create rga buffer pools
  rgb_buffer_pool_ = std::make_unique<RgaBufferPool>(kMaxDecodedFrameBufferCount, RK_FORMAT_BGR_888,
                                                     kInputWidth, kInputHeight);
  scale_buffer_pool_ = std::make_unique<RgaBufferPool>(
      kMaxDecodedFrameBufferCount, RK_FORMAT_YCbCr_420_SP, kInputWidth, kInputHeight);
#endif

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

#if TEST_WITH_IMAGE
// Load input data from binary file
int App::LoadTestData(const std::string& bin_path, std::vector<float>& input_data) {
  // Read entire input
  std::vector<uint8_t> bgr(kInputTensorSize);
  if (!read_file(bin_path, bgr)) {
    LOGE(TAG, "Failed to read input file: %s", bin_path.c_str());
    return -1;
  }
  // Convert BGR24 to NHWC float32
  bgr24_to_nhwc_float(bgr.data(), kInputWidth, kInputHeight, input_data.data(), 1.0f);

  return 0;
}

// Load input data from image file(backend is OpenCV)
int App::LoadTestImage(const std::string& img_path, std::vector<float>& input_data) {
  cv::Mat img = cv::imread(img_path);
  if (img.empty()) {
    LOGE(TAG, "Failed to read image: %s", img_path.c_str());
    return -1;
  }

  // Convert to NHWC float32
  bgr_to_nhwc_float_no_norm(img, input_data);

  return 0;
}
#endif

int App::Inference(const rga_buffer_t* buffer) {
#if TEST_WITH_IMAGE
  if (!input_data_ready) {
    LOGE(TAG, "Input data is not ready");
    return -1;
  }
#endif

#if 0
  cv::Mat bgr_image(kInputHeight, MPP_ALIGN(kInputWidth, 16), CV_8UC3, buffer->vir_addr);
  if (!cv::imwrite(BGR_FILE_PATH, bgr_image)) {
    LOGE(TAG, "Failed to write BGR image to file: %s", BGR_FILE_PATH);
  }
#endif
  // cv::Mat bgr_image(kInputHeight, MPP_ALIGN(kInputWidth, 16), CV_8UC3, buffer->vir_addr);
  // bgr_to_nhwc_float_no_norm(bgr_image, input_data);

  int ret = 0;

  // Prepare input
#if USE_ZEROS_COPY
  // Copy input data to input tensor memory
  rknn_tensor_mem* mem = input_mems[0];
  if (!mem) {
    LOGE(TAG, "input mem is null");
    return -1;
  }
  // Convert BGR24 to NHWC float32
  bgr24_to_nhwc_float_stride((uint8_t*) buffer->vir_addr, kInputWidth, kInputHeight,
                             MPP_ALIGN(kInputWidth, 16), (float*) mem->virt_addr, 1.0f);
#else

  // Convert BGR24 to NHWC float32
  bgr24_to_nhwc_float_stride((uint8_t*) buffer->vir_addr, kInputWidth, kInputHeight,
                             MPP_ALIGN(kInputWidth, 16), input_data.data(), 1.0f);

  // Set input
  std::vector<rknn_input> inputs(io_num.n_input);
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_FLOAT32;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size = kInputTensorSizeInBytes;
  inputs[0].buf = input_data.data();

  ret = rknn_inputs_set(rknn_ctx, io_num.n_input, inputs.data());
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "failed to set input, ret=%d", ret);
    return false;
  }
#endif

  // Run inference
  ret = rknn_run(rknn_ctx, nullptr);
  if (ret != RKNN_SUCC) {
    LOGE(TAG, "rknn_run fail ret=%d", ret);
    return ret;
  }

  LOGI(TAG, "Inference completed successfully");

#if USE_ZEROS_COPY
  // Bounding boxes
  const float* bboxes = reinterpret_cast<const float*>(output_mems[0]->virt_addr);
  // Bounding box scores
  const float* bbox_scores = reinterpret_cast<const float*>(output_mems[1]->virt_addr);
  // Anchors, order: [4,2,3,5]
  const float* anchors[NUM_ANCHORS] = {
      reinterpret_cast<const float*>(output_mems[4]->virt_addr),
      reinterpret_cast<const float*>(output_mems[2]->virt_addr),
      reinterpret_cast<const float*>(output_mems[3]->virt_addr),
      reinterpret_cast<const float*>(output_mems[5]->virt_addr),
  };

  // Post-process output
  PostProcess(bboxes, bbox_scores, anchors);
#else
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

  // Post-process output
  PostProcess(bboxes, bbox_scores, anchors);
#endif

  return ret;
}

// index=0, name=bboxes, n_dims=2, dims=[1, 17200], n_elems=17200, size=34400
// index=1, name=bboxes_scores, n_dims=2, dims=[1, 8600], n_elems=8600, size=17200
// index=2, name=anchor3, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
// index=3, name=anchor2, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
// index=4, name=anchor1, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
// index=5, name=anchor4, n_dims=4, dims=[1, 25, 43, 3], n_elems=3225, size=6450
void App::PostProcess(const float* bboxes, const float* bbox_scores, const float* const* anchors) {
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
    } else {
      LOGD(TAG, "Detected action: %s, conf: %.3f", ACTION_CLASS_LIST[action_label], action_conf);
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

void App::OnDecodedFrame(const VideoFrameSlot& frame) {
  if (frame.eos) {
    ready_.store(false);
    LOGW(TAG, "detect eos");
    return;
  }

  // save to ring buffer
  ring_buffer_->Enqueue(frame);

  // check need refresh
  if (frame.need_refresh) {
    LOGW(TAG, "need refresh");
    ready_.store(false);
    // release all imported rga buffers
    for (auto& [key, buffer] : rga_buffers_) {
      releasebuffer_handle(buffer->handle);
    }
    rga_buffers_.clear();
  }

  auto count = rga_buffers_.count(frame.fd);
  if (count == 0) {
    auto buffer = std::make_unique<rga_buffer_t>();
    if (ImportRgaBuffer(frame, buffer.get())) {
      rga_buffers_[frame.fd] = std::move(buffer);
    }

    if (rga_buffers_.size() == kMaxDecodedFrameBufferCount) {
      last_success_fd_ = frame.fd;

      ready_.store(true);
    }
  }
}

bool App::ImportRgaBuffer(const VideoFrameSlot& frame, rga_buffer_t* buffer) {
  im_handle_param_t handle_param;
  handle_param.width = frame.width;
  handle_param.height = frame.height;
  handle_param.format = RK_FORMAT_YCbCr_420_SP;  // NV12

  // get rga buffer handle from decoded buffer
  rga_buffer_handle_t handle = importbuffer_fd(frame.fd, &handle_param);
  if (handle <= 0) {
    LOGE(TAG, "rga import dma buffer failed, fd=%d", frame.fd);
    return false;
  }

  *buffer =
      wrapbuffer_handle(handle, (int) frame.width, (int) frame.height, (int) handle_param.format,
                        (int) frame.width_stride, (int) frame.height_stride);
  buffer->fd = frame.fd;

  return true;
}

rga_buffer_t* App::GetImportedRgaBuffer() {
  VideoFrameSlot frame = {0};
  bool success = ring_buffer_->Dequeue(&frame);
  if (success) {
    last_success_fd_ = frame.fd;
  }
  if (rga_buffers_.count(last_success_fd_) == 0) {
    return nullptr;
  }
  return rga_buffers_.at(last_success_fd_).get();
}

rga_buffer_t* App::ScaleConvert(rga_buffer_t* src) {
  rga_buffer_t* scale_dst = scale_buffer_pool_->Acquire();
  if (scale_dst == nullptr) {
    LOGE(TAG, "failed to acquire rga buffer");
    return nullptr;
  }

  im_rect rect = {0, 0, (int) kInputWidth, (int) kInputHeight};
  auto ret = imcheck(*src, *scale_dst, {}, rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "scale to file check failed, %s", imStrError(ret));
    return nullptr;
  }

  ret = improcess(*src, *scale_dst, {}, {}, rect, {}, IM_SYNC);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "scale to file failed, %s", imStrError(ret));
    return nullptr;
  }

  rga_buffer_t* dst = rgb_buffer_pool_->Acquire();
  if (dst == nullptr) {
    LOGE(TAG, "failed to acquire rga buffer");
    return nullptr;
  }

  ret = imcvtcolor(*scale_dst, *dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "convert to rgb888 failed, %s", imStrError(ret));
    return nullptr;
  }

  return dst;
}

bool App::Start() {
  if (running.load()) {
    LOGW(TAG, "App already running");
    return false;
  }

  if (!capturer) {
    capturer = std::make_unique<VideoFileReader>(TEST_VID_FILE_PATH, this);
    if (!capturer->Start()) {
      LOGE(TAG, "Failed to start video capturer");
      return false;
    }
  }

  running.store(true);
  worker = std::make_unique<std::thread>(&App::MainLoop, this);
  return true;
}

void App::Stop() {
  if (!running.load()) {
    LOGW(TAG, "App not running");
    return;
  }

  if (capturer) {
    capturer->Stop();
    capturer.reset();
  }

  running.store(false);
  if (worker && worker->joinable()) {
    worker->join();
    worker.reset();
  }
}

void App::MainLoop() {
  LOGI(TAG, "App main loop started");

  while (running.load()) {
    if (!ready_.load()) {
      // frames are not ready for use, wait for a while
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    auto src_buffer = GetImportedRgaBuffer();
    if (src_buffer == nullptr) {
      LOGE(TAG, "failed to get imported rga buffer");
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // do letter boxing
    auto dest_buffer = ScaleConvert(src_buffer);
    if (dest_buffer == nullptr) {
      LOGE(TAG, "failed to do letter boxing");
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // do inference
    Inference(dest_buffer);
  }

  LOGI(TAG, "App main loop exited");
}

/////////////////////////////////////////////////////////////////////////////////
std::unique_ptr<App> app_instance;  // Global app instance

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
  if (!std::filesystem::exists(TEST_BIN_FILE_PATH)) {
    LOGE(TAG, "Test file not found: %s", TEST_BIN_FILE_PATH);
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

  if (!app_instance->Start()) {
    LOGE(TAG, "Failed to start application");
    return -1;
  }

  // Main loop
  while (app_instance->running.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Release resources
  app_instance->Stop();

  app_instance.reset();

  return 0;
}