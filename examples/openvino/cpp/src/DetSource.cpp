#include "DetSource.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "Common.h"

#define MODEL_PATH "/data/local/tmp/lldb-standalone/hand_sign_v8n.rknn"

#define TAG "DetSource"

namespace det {

constexpr uint32_t kInputWidth = 640;
constexpr uint32_t kInputHeight = 640;
constexpr uint32_t kMaxValidBBoxes = 32;
constexpr size_t kInputTensorSize = 1 * kInputHeight * kInputWidth * 3;
constexpr size_t kInputTensorSizeInBytes = kInputTensorSize * sizeof(float);
constexpr float kDetModelClassScoreThreshold = 0.85f;  // Score threshold for valid detections
constexpr float kDetModelNmsThreshold = 0.4f;          // Non-Maximum Suppression threshold

constexpr const char* kClassNames[kNumOfClasses] = {"one",  "two",  "three", "four",
                                                    "five", "good", "ok"};

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

// Convert RGB24 to NHWC float32 with stride support
static void rgb24_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride, float* dst,
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
static void rgba32_to_nhwc_float_stride(const uint8_t* src, int W, int H, int stride, float* dst,
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

// Convert RGB24 to NHWC float32
static void rgb24_to_nhwc_float(const uint8_t* src, int W, int H, float* dst, float scale) {
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

// Simple word-wrap using Hershey metrics.
static std::vector<std::string> wrap_text(const std::string& text, int max_width_px, int font_face,
                                          double font_scale, int thickness) {
  if (max_width_px <= 0) {
    // Split only on '\n'.
    std::vector<std::string> lines;
    size_t start = 0;
    while (true) {
      size_t p = text.find('\n', start);
      if (p == std::string::npos) {
        lines.push_back(text.substr(start));
        break;
      }
      lines.push_back(text.substr(start, p - start));
      start = p + 1;
    }
    if (lines.empty()) lines.push_back("");
    return lines;
  }

  std::vector<std::string> lines;
  std::string current;
  std::string word;

  auto Measure = [&](const std::string& s) -> int {
    int base = 0;
    return cv::getTextSize(s, font_face, font_scale, thickness, &base).width;
  };

  for (size_t i = 0; i <= text.size(); ++i) {
    const char c = (i < text.size()) ? text[i] : ' ';
    if (c == ' ' || c == '\n' || i == text.size()) {
      if (!word.empty()) {
        const std::string candidate = current.empty() ? word : (current + " " + word);
        if (Measure(candidate) > max_width_px && !current.empty()) {
          lines.push_back(current);
          current = word;
        } else {
          current = candidate;
        }
        word.clear();
      }
      if (c == '\n') {
        lines.push_back(current);
        current.clear();
      }
    } else {
      word.push_back(c);
    }
  }
  if (!current.empty()) lines.push_back(current);
  if (lines.empty()) lines.push_back("");
  return lines;
}

static void render_text_to_raw_rgba(const std::string& text, cv::Mat* out_mat, int* out_w,
                                    int* out_h, size_t* out_stride, int font_face,
                                    double font_scale, int thickness, cv::Scalar color_rgba,
                                    int padding, int line_spacing, int max_width_px,
                                    bool add_shadow) {
  if (out_mat == nullptr || out_w == nullptr || out_h == nullptr || out_stride == nullptr) {
    return;
  }

  // 1) Wrap and measure.
  const std::vector<std::string> lines =
      wrap_text(text, max_width_px, font_face, font_scale, thickness);

  int base = 0;
  int max_w = 0;
  int total_h = 0;
  std::vector<cv::Size> sizes;
  sizes.reserve(lines.size());

  for (const std::string& ln : lines) {
    const cv::Size sz = cv::getTextSize(ln, font_face, font_scale, thickness, &base);
    sizes.push_back(sz);
    if (sz.width > max_w) max_w = sz.width;
    total_h += sz.height + line_spacing;
  }
  if (!lines.empty()) total_h -= line_spacing;

  const int width = std::max(1, max_w + padding * 2);
  const int height = std::max(1, total_h + padding * 2);

  // 2) Alpha mask (8-bit) for coverage.
  cv::Mat alpha(height, width, CV_8UC1, cv::Scalar(0));

  if (add_shadow) {
    const int kShadowDx = 1;
    const int kShadowDy = 1;
    const int kShadowAlpha = 200;
    int y = padding;
    for (size_t i = 0; i < lines.size(); ++i) {
      const int yline = y + sizes[i].height;
      cv::putText(alpha, lines[i], cv::Point(padding + kShadowDx, yline + kShadowDy), font_face,
                  font_scale, kShadowAlpha, thickness + 2, cv::LINE_AA);
      y += sizes[i].height + line_spacing;
    }
  }

  {
    int y = padding;
    for (size_t i = 0; i < lines.size(); ++i) {
      const int yline = y + sizes[i].height;
      cv::putText(alpha, lines[i], cv::Point(padding, yline), font_face, font_scale, 255, thickness,
                  cv::LINE_AA);
      y += sizes[i].height + line_spacing;
    }
  }

  // 3) Build RGBA in **RGBA** order.
  // OpenCV cv::Scalar stores (val0, val1, val2, val3) = (R, G, B, A) as given.
  cv::Mat r(height, width, CV_8UC1, cv::Scalar(static_cast<uint8_t>(color_rgba[2])));
  cv::Mat g(height, width, CV_8UC1, cv::Scalar(static_cast<uint8_t>(color_rgba[1])));
  cv::Mat b(height, width, CV_8UC1, cv::Scalar(static_cast<uint8_t>(color_rgba[0])));
  cv::Mat a(height, width, CV_8UC1, cv::Scalar(0));

  if (static_cast<int>(color_rgba[3]) == 255) {
    alpha.copyTo(a);
  } else {
    cv::Mat scaled;
    alpha.convertTo(scaled, CV_32F, color_rgba[3] / 255.0);
    scaled.convertTo(a, CV_8U);
  }

  // Zero-out then mask-copy color where alpha > 0.
  r.setTo(0);
  g.setTo(0);
  b.setTo(0);

  cv::Mat tmp(height, width, CV_8UC1, cv::Scalar(static_cast<uint8_t>(color_rgba[2])));
  tmp.copyTo(r, alpha);
  tmp.setTo(static_cast<uint8_t>(color_rgba[1]));
  tmp.copyTo(g, alpha);
  tmp.setTo(static_cast<uint8_t>(color_rgba[0]));
  tmp.copyTo(b, alpha);

  std::vector<cv::Mat> channels;
  channels.reserve(4);
  channels.push_back(r);
  channels.push_back(g);
  channels.push_back(b);
  channels.push_back(a);

  cv::merge(channels, *out_mat);  // CV_8UC4, RGBA order.

  // 4) Ensure contiguous and export raw bytes.
  if (!out_mat->isContinuous()) {
    *out_mat = out_mat->clone();
  }
  *out_stride = static_cast<size_t>(out_mat->step[0]);
  *out_w = out_mat->cols;
  *out_h = out_mat->rows;
}

OsdText::OsdText(const std::string& text) : text_(text), w_(0), h_(0), stride_(0) {
  // Constants for text rendering
  const int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  const double kFontScale = 1.0;
  const int kThickness = 2;
  const cv::Scalar kColorRgba(255, 255, 255, 255);  // White, opaque
  const int kPadding = 12;
  const int kLineSpacing = 6;
  const int kMaxWidthPx = 480;

  memset(&config_, 0, sizeof(im_osd_t));
  memset(&rga_buffer_, 0, sizeof(rga_buffer_t));

  // Calculate RGBA image size and render text to RGBA buffer
  render_text_to_raw_rgba(text_, &rgba_, &w_, &h_, &stride_, kFontFace, kFontScale, kThickness,
                          kColorRgba, kPadding, kLineSpacing, kMaxWidthPx, false);

  // Configure OSD settings
  config_.block_parm.width_mode = IM_OSD_BLOCK_MODE_NORMAL;
  config_.block_parm.width = w_;
  config_.block_parm.block_count = 1;
  config_.block_parm.background_config = IM_OSD_BACKGROUND_DEFAULT_BRIGHT;
  config_.block_parm.direction = IM_OSD_MODE_HORIZONTAL;
  config_.block_parm.color_mode = IM_OSD_COLOR_PIXEL;

  config_.invert_config.invert_channel = IM_OSD_INVERT_CHANNEL_COLOR;
  config_.invert_config.flags_mode = IM_OSD_FLAGS_EXTERNAL;
  config_.invert_config.invert_flags = 0x000000000000002a;
  config_.invert_config.flags_index = 1;
  config_.invert_config.threash = 40;
  config_.invert_config.invert_mode = IM_OSD_INVERT_USE_SWAP;

  // Allocate DMA buffer for RGA
  char* vir_addr = nullptr;
  auto buf_size = static_cast<size_t>(stride_) * static_cast<size_t>(h_);
  int fd = -1;
  int ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, buf_size, &fd, (void**) &vir_addr);
  if (ret != 0) {
    assert(false && "OsdText::OsdText: alloc dma32_heap buffer failed");
  }

  // Copy the rendered RGBA data to the allocated buffer
  memcpy(vir_addr, rgba_.data, buf_size);

  // Import to RGA buffer
  auto handle = importbuffer_fd(fd, buf_size);
  if (handle == 0) {
    assert(false && "OsdText::OsdText: importbuffer_fd failed");
  }

  rga_buffer_ = wrapbuffer_handle(handle, w_, h_, RK_FORMAT_RGBA_8888);
  rga_buffer_.fd = fd;  // Keep the fd for later free
}

OsdText::~OsdText() {
  if (rga_buffer_.handle > 0) {
    releasebuffer_handle(rga_buffer_.handle);
    rga_buffer_.handle = 0;
  }
  if (rga_buffer_.fd > 0) {
    auto buf_size = static_cast<size_t>(stride_) * static_cast<size_t>(h_);
    dma_buf_free(buf_size, &rga_buffer_.fd, rga_buffer_.vir_addr);
    rga_buffer_.fd = -1;
  }
}

DetSource::DetSource(std::string_view id)
    : id_(id),
      running_(false),
      worker_thread_(nullptr),
      rknn_ctx_(0),
      last_success_fd_(-1),
      ready_(false),
      frame_width_(1920),
      frame_height_(1080) {
  LOGI(TAG, "Create DetSource with id: %s", id_.c_str());
  auto ret = Init();
  assert(ret == 0 && "DetSource Init failed");
}

DetSource::~DetSource() {
  LOGI(TAG, "Destroying DetSource with id: %s", id_.c_str());
  DeInit();
  LOGI(TAG, "DetSource with id: %s destroyed", id_.c_str());
}

int DetSource::Init() {
  int ret = 0;

  // Load model
  uint32_t model_len = 0;
  auto model = load_model(MODEL_PATH, &model_len);
  if (model == nullptr) {
    LOGE(TAG, "load model failed");
    return -1;
  }

  // Init RKNN
  ret = rknn_init(&rknn_ctx_, model, model_len, 0, nullptr);
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
  input_attrs_.reserve(io_num_.n_input);
  for (int i = 0; i < io_num_.n_input; i++) {
    input_attrs_[i].index = i;  // Set index for input tensor
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_query input attr fail ret=%d", ret);
      return -1;
    }

    dump_tensor_attr(&input_attrs_[i]);
  }

  // Query output tensor shape
  output_attrs_.reserve(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; i++) {
    output_attrs_[i].index = i;  // Set index for output tensor
    ret =
        rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_query output attr fail ret=%d", ret);
      return -1;
    }

    dump_tensor_attr(&output_attrs_[i]);
  }

  // Create input tensors
  input_mems_.reserve(io_num_.n_input);
  for (int i = 0; i < io_num_.n_input; i++) {
    input_attrs_[i].fmt = RKNN_TENSOR_NHWC;      // Set format to NHWC
    input_attrs_[i].type = RKNN_TENSOR_FLOAT32;  // float32, not quantized
    input_attrs_[i].size = kInputTensorSizeInBytes;
    input_attrs_[i].size_with_stride = kInputTensorSizeInBytes;

    // Create memory for this tensor
    auto* mem = rknn_create_mem(rknn_ctx_, kInputTensorSizeInBytes);
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

  // Create output tensors: [1, 11, 8400]
  output_mems_.reserve(io_num_.n_output);
  for (int i = 0; i < io_num_.n_output; i++) {
    output_attrs_[i].type = RKNN_TENSOR_FLOAT32;  // float32, not quantized

    // Calculate output tensor size
    output_attrs_[i].size = output_attrs_[i].dims[1] * output_attrs_[i].dims[2] * sizeof(float);
    output_attrs_[i].size_with_stride = output_attrs_[i].size;

    // Create memory for this tensor
    auto* mem = rknn_create_mem(rknn_ctx_, output_attrs_[i].size);
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

  // create ring buffer
  ring_buffer_ = std::make_unique<RingBuffer<VideoFrameSlot> >(kDefaultRingBufferSize);
  // create rga buffer pools
  rgb_buffer_pool_ = std::make_unique<RgaBufferPool>(
      kMaxDecodedFrameBufferCount, RK_FORMAT_RGBA_8888, frame_width_, frame_height_);
  nv12_buffer_pool_ = std::make_unique<RgaBufferPool>(
      kMaxDecodedFrameBufferCount, RK_FORMAT_YCbCr_420_SP, frame_width_, frame_height_);
  scale_buffer_pool_ = std::make_unique<RgaBufferPool>(
      kMaxDecodedFrameBufferCount, RK_FORMAT_RGBA_8888, kInputWidth, kInputHeight);

  // create osd texts
  CreateOsdTexts();

  return 0;
}

void DetSource::DeInit() {
  // Stop first
  Stop();

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

  // Free OSD texts
  DestroyOsdTexts();

  LOGI(TAG, "DetSource deinitialized successfully");
}

void DetSource::CreateOsdTexts() {
  for (size_t i = 0; i < kNumOfClasses; ++i) {
    auto osd_text = std::make_unique<OsdText>(kClassNames[i]);
    osd_texts_.emplace_back(std::move(osd_text));
  }
}

void DetSource::DestroyOsdTexts() {
  for (auto& osd_text : osd_texts_) {
    osd_text.reset();
  }
  osd_texts_.clear();
}

bool DetSource::Start() {
  if (running_.load()) {
    LOGW(TAG, "DetSource is already running");
    return false;
  }

  running_.store(true);
  worker_thread_ = std::make_unique<std::thread>(&DetSource::MainLoop, this);
  LOGI(TAG, "DetSource started");
  return true;
}

void DetSource::Stop() {
  if (!running_.load()) {
    LOGW(TAG, "DetSource is not running");
    return;
  }

  running_.store(false);
  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }
  worker_thread_.reset();
  LOGI(TAG, "DetSource stopped");
}

void DetSource::OnVideoFrame(const VideoFrameSlot& frame) {
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

      frame_width_ = frame.width;
      frame_height_ = frame.height;
      // calculate letter box
      CalculateLetterBox();

      ready_.store(true);
    }
  }
}

void DetSource::AddVideoSink(VideoSink* sink) {
  std::lock_guard<std::mutex> lock(sink_mutex_);

  if (std::find(sinks_.begin(), sinks_.end(), sink) != sinks_.end()) {
    LOGW(TAG, "Sink already added");
    return;
  }

  sinks_.push_back(sink);
  LOGI(TAG, "Added video sink, total sinks: %zu", sinks_.size());
}

void DetSource::RemoveVideoSink(VideoSink* sink) {
  std::lock_guard<std::mutex> lock(sink_mutex_);

  if (std::find(sinks_.begin(), sinks_.end(), sink) == sinks_.end()) {
    LOGW(TAG, "Sink not found");
    return;
  }

  sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
  LOGI(TAG, "Removed video sink, total sinks: %zu", sinks_.size());
}

bool DetSource::ImportRgaBuffer(const VideoFrameSlot& frame, rga_buffer_t* buffer) {
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

rga_buffer_t* DetSource::GetImportedRgaBuffer() {
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

rga_buffer_t* DetSource::Convert2RGBA(rga_buffer_t* src) {
  rga_buffer_t* dst = rgb_buffer_pool_->Acquire();
  if (dst == nullptr) {
    LOGE(TAG, "failed to acquire rga buffer");
    return nullptr;
  }

  auto ret = imcvtcolor(*src, *dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGBA_8888);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "convert to rgb888 failed, %s", imStrError(ret));
    return nullptr;
  }

  return dst;
}

rga_buffer_t* DetSource::Convert2NV12(rga_buffer_t* src) {
  rga_buffer_t* dst = nv12_buffer_pool_->Acquire();
  if (dst == nullptr) {
    LOGE(TAG, "failed to acquire rga buffer");
    return nullptr;
  }

  auto ret = imcvtcolor(*src, *dst, RK_FORMAT_RGBA_8888, RK_FORMAT_YCbCr_420_SP);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "convert to rgb888 failed, %s", imStrError(ret));
    return nullptr;
  }

  return dst;
}

rga_buffer_t* DetSource::Letterbox(rga_buffer_t* src) {
  auto width = (float) src->width * letter_box_.scale;
  auto height = (float) src->height * letter_box_.scale;

  rga_buffer_t* scale_dst = scale_buffer_pool_->Acquire();
  if (scale_dst == nullptr) {
    LOGE(TAG, "failed to acquire rga buffer");
    return nullptr;
  }

  im_rect rect = {letter_box_.x_pad, letter_box_.y_pad, (int) width, (int) height};

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

  return scale_dst;
}

void DetSource::MainLoop() {
  LOGI(TAG, "DetSource main loop started");

  while (running_.load()) {
    if (!ready_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // auto begin = std::chrono::steady_clock::now();

    auto src_buffer = GetImportedRgaBuffer();
    if (src_buffer == nullptr) {
      LOGE(TAG, "failed to get imported rga buffer");
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // Convert to RGBA32
    rga_buffer_t* rgb_buffer = Convert2RGBA(src_buffer);
    if (rgb_buffer == nullptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // Letterbox
    rga_buffer_t* scaled_buffer = Letterbox(rgb_buffer);
    if (scaled_buffer == nullptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // Copy input data to input tensor memory
    rknn_tensor_mem* mem = input_mems_[0];
    if (!mem) {
      LOGE(TAG, "input mem is null");
      return;
    }
    // Convert RGB24 to NHWC float32
    rgba32_to_nhwc_float_stride((uint8_t*) scaled_buffer->vir_addr, kInputWidth, kInputHeight,
                                (int) scaled_buffer->wstride, (float*) mem->virt_addr,
                                1.0f / 255.0f);

    // Inference
    auto ret = rknn_run(rknn_ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_run fail ret=%d", ret);
      continue;
    }

    // Post-process
    auto detected_objects = PostProcess();
    for (const auto& obj : detected_objects) {
      LOGI(TAG, "Detected: %s, score=%.3f, box=[%d, %d, %d, %d]", kClassNames[obj.label], obj.score,
           obj.box.x, obj.box.y, obj.box.width, obj.box.height);
    }

    // Draw OSD
    DrawOsd(rgb_buffer, detected_objects);

    // Convert back to NV12
    rga_buffer_t* nv12_buffer = Convert2NV12(rgb_buffer);

    {
      // Callbacks to sinks
      std::lock_guard<std::mutex> lock(sink_mutex_);
      current_frame_ = {(RK_U32) nv12_buffer->width,
                        (RK_U32) nv12_buffer->height,
                        (RK_U32) nv12_buffer->wstride,
                        (RK_U32) nv12_buffer->hstride,
                        MPP_FMT_YUV420SP,
                        nv12_buffer->fd,
                        false,
                        false};
      for (auto* sink : sinks_) {
        sink->OnVideoFrame(current_frame_);
      }
    }

    // auto end = std::chrono::steady_clock::now();
    // auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    // LOGD(TAG, "Processed one frame in %llu ms", elapsed_ms);
  }

  LOGD(TAG, "DetSource main loop exited");
}

void DetSource::CalculateLetterBox() {
  auto original_width = (float) frame_width_;
  auto original_height = (float) frame_height_;
  // calculate the scale
  float scale =
      std::min((float) kInputWidth / original_width, (float) kInputWidth / original_height);

  // calculate the scaled image dimensions
  float scaled_width = original_width * scale;
  float scaled_height = original_height * scale;

  // calculate padding
  int x_pad = static_cast<int>(((float) kInputWidth - scaled_width) / 2);
  int y_pad = static_cast<int>(((float) kInputWidth - scaled_height) / 2);

  letter_box_.x_pad = x_pad;
  letter_box_.y_pad = y_pad;
  letter_box_.scale = scale;
}

cv::Rect DetSource::Unletterbox(float cx, float cy, float w, float h) {
  // Convert to: x0, y0, x1, y1
  float x0 = cx - w * 0.5f;
  float y0 = cy - h * 0.5f;
  float x1 = cx + w * 0.5f;
  float y1 = cy + h * 0.5f;

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

std::vector<DetectedObject> DetSource::PostProcess() {
  const float* output = (float*) output_mems_[0]->virt_addr;

  // Output shape: [1, 11, 8400] => [1, C, N] where C=4+kNumOfClasses, N=num candidates
  const uint32_t C = output_attrs_[0].dims[1];
  const uint32_t N = output_attrs_[0].dims[2];
  const uint32_t expectedC = 4 + kNumOfClasses;
  if (C != expectedC) {
    LOGE(TAG, "Unexpected channel size C=%u, expected %u (4 + num_classes)", C, expectedC);
    return {};
  }

  auto at = [&](uint32_t c, uint32_t n) -> float {
    // Layout is [C, N] contiguous
    return output[n + N * c];
  };

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (uint32_t i = 0; i < N; ++i) {
    // Argmax over class channels [4 .. 4+kNumOfClasses)
    float best_cls_score = -std::numeric_limits<float>::infinity();
    int best_cls_id = -1;
    for (uint32_t j = 0; j < kNumOfClasses; ++j) {
      float s = at(4 + j, i);
      if (s > best_cls_score) {
        best_cls_score = s;
        best_cls_id = static_cast<int>(j);
      }
    }

    if (best_cls_id < 0) continue;

    // If the model outputs logits, consider applying sigmoid here.
    float final_score = best_cls_score;
    if (final_score > kDetModelClassScoreThreshold) {
      float cx = at(0, i);
      float cy = at(1, i);
      float w = at(2, i);
      float h = at(3, i);

      if (!(w > 0.0f && h > 0.0f)) continue;

      scores.push_back(final_score);
      class_ids.push_back(best_cls_id);
      boxes.emplace_back(Unletterbox(cx, cy, w, h));
    }
  }

  if (boxes.empty()) {
    // LOGW(TAG, "No valid boxes found after thresholding");
    return {};  // No valid boxes to process
  }

  // Apply Non-Maximum Suppression (NMS)
  std::vector<int> indices;
  indices.reserve(kMaxValidBBoxes);
  cv::dnn::NMSBoxes(boxes, scores, kDetModelClassScoreThreshold, kDetModelNmsThreshold, indices,
                    1.f, kMaxValidBBoxes * 2);

  std::vector<DetectedObject> detected_objects;
  detected_objects.reserve(indices.size());
  for (int idx : indices) {
    detected_objects.emplace_back(boxes[idx], class_ids[idx], scores[idx]);
  }

  return detected_objects;
}

void DetSource::DrawOsd(rga_buffer_t* buffer, const std::vector<DetectedObject>& objects) {
  if (objects.empty()) {
    return;
  }

  std::vector<im_rect> rects;
  rects.reserve(objects.size());

  im_job_handle_t job = imbeginJob();

  int ret = 0;
  for (const auto& obj : objects) {
    int x, y, w, h;
    x = obj.box.x & (~1);       // x must be even
    y = obj.box.y & (~1);       // y must be even
    w = obj.box.width & (~1);   // w must be even
    h = obj.box.height & (~1);  // h must be even

    // Draw box
    im_rect rect{x, y, w, h};
    ret = imcheck({}, *buffer, {}, rect, IM_COLOR_FILL);
    if (IM_STATUS_NOERROR != ret) {
      LOGE(TAG, "draw rectangle check failed, %s", imStrError(ret));
      continue;
    }
    ret = imrectangleTask(job, *buffer, rect, 0xff00ff00, 2);
    if (IM_STATUS_SUCCESS != ret) {
      LOGE(TAG, "apply draw rectangle task failed, %s", imStrError(ret));
      continue;
    }

    // Draw label
    const auto& osd = osd_texts_[obj.label];
    int osd_y = std::max(0, y - osd->GetHeight() - 2);
    im_rect osd_rect = {x, osd_y, osd->GetWidth(), osd->GetHeight()};
    auto& osd_img = osd->GetRgaBuffer();
    ret = imcheck(osd_img, *buffer, {}, osd_rect);
    if (IM_STATUS_NOERROR != ret) {
      LOGE(TAG, "draw label check failed, %s", imStrError(ret));
      continue;
    }
    ret = imosdTask(job, osd_img, *buffer, osd_rect, osd->GetOsdConfigPtr());
    if (IM_STATUS_SUCCESS != ret) {
      LOGE(TAG, "apply draw label task failed, %s", imStrError(ret));
      continue;
    }
  }

  ret = imendJob(job);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "imendJob failed, %s", imStrError(ret));
    imcancelJob(job);
  }
}

}  // namespace det