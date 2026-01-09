#include "DetSource.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <latch>
#include <limits>
#include <numeric>
#include <set>

#include "Common.h"
#include "utils/Utils.h"

#if USE_QUANTIZED_MODEL
#define DET_MODEL_PATH "/data/local/tmp/lldb-standalone/det_500m_pruned_i8.rknn"
#define EXT_MODEL_PATH "/data/local/tmp/lldb-standalone/w600k_mbf_fixed_i8.rknn"
// #define DET_MODEL_PATH "/data/local/tmp/lldb-standalone/det_2.5g_fixed_i8.rknn"
// #define EXT_MODEL_PATH "/data/local/tmp/lldb-standalone/w600k_r50_fixed_i8.rknn"
#else
#define DET_MODEL_PATH "/data/local/tmp/lldb-standalone/det_500m_fixed_f32.rknn"
#define EXT_MODEL_PATH "/data/local/tmp/lldb-standalone/w600k_mbf_fixed_f32.rknn"
#endif

#define TAG "DetSource"

static uint32_t infer_count = 0;
static uint64_t total_infer_time = 0;

namespace det {

constexpr uint32_t kOsdColor = 0xff00ff00;
constexpr float kDetScoreThreshold = 0.5f;
constexpr float kDetNmsThreshold = 0.4f;
constexpr float kRecogSimilarityThreshold = 0.45f;
constexpr int kExtractorInputSize = 112;

// Cosine similarity for face matching
static float compare_faces(const face::Embeddings& emb1, const face::Embeddings& emb2) {
  float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
  for (size_t i = 0; i < emb1.size(); i++) {
    dot += emb1[i] * emb2[i];
    norm1 += emb1[i] * emb1[i];
    norm2 += emb2[i] * emb2[i];
  }
  return dot / (sqrt(norm1) * sqrt(norm2));
}

// Word-wrap using Hershey metrics.
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

DetSource::DetSource(std::string_view id, const std::vector<face::FaceLocation>& register_faces)
    : id_(id),
      running_(false),
      worker_thread_(nullptr),
      thread_pool_(nullptr),
      extractors_pool_(nullptr),
      detector_(nullptr),
      last_success_fd_(-1),
      ready_(false),
      frame_width_(1920),
      frame_height_(1080),
      registered_faces_(register_faces) {
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
  // Create detector
  detector_ =
      std::make_unique<face::FaceDetector>(DET_MODEL_PATH, kDetScoreThreshold, kDetNmsThreshold);
#if PARALLEL_EXECUTION
  thread_pool_ =
      std::make_unique<ThreadPool>(/*num_threads=*/MAX_PARALLEL_TASKS, /*max_queue_size=*/64);
  extractors_pool_ = std::make_unique<ContextPool>(MAX_PARALLEL_TASKS);
  // Create extractors
  auto main_extractor = new face::FaceExtractorP(EXT_MODEL_PATH, frame_width_, frame_height_);
  extractors_.push_back(main_extractor);
  rknn_context main_ctx = main_extractor->GetRknnContext();
  for (int i = 1; i < MAX_PARALLEL_TASKS; ++i) {
    auto extractor = new face::FaceExtractorP(&main_ctx, frame_width_, frame_height_);
    extractors_.push_back(extractor);
  }
  CreateCropBuffers();
#else
  // Create extractor
  extractor_ =
      std::make_unique<face::FaceExtractor>(EXT_MODEL_PATH, false, frame_width_, frame_height_);
#endif  // PARALLEL_EXECUTION
  // Create ring buffer
  ring_buffer_ = std::make_unique<RingBuffer<VideoFrameSlot>>(kDefaultRingBufferSize);
  // Create rga buffers
  CreateRgaBuffers();
  // Create OSD texts
  CreateOsdTexts();

  return 0;
}

void DetSource::DeInit() {
  // Stop first
  Stop();

  // Release RGA buffers
  DestroyRgaBuffers();

  // Destroy OSD texts
  DestroyOsdTexts();

  // Release detector
  detector_.reset();

#if PARALLEL_EXECUTION
  // Release extractors
  for (auto& extractor : extractors_) {
    delete extractor;
  }
  extractors_.clear();
  crop_buffers_.clear();
  DestroyCropBuffers();
#else
  // Release extractor
  extractor_.reset();
#endif  // PARALLEL_EXECUTION

  LOGI(TAG, "DetSource deinitialized successfully");
}

void DetSource::CreateRgaBuffers() {
  int ret = 0;
  // RGB24 buffer for input: NV12 -> RGB24
  ret = util::rga::create_rga_buffer(frame_width_, frame_height_, RK_FORMAT_RGB_888, rgb_buffer_);
  if (ret != 0) {
    LOGE(TAG, "create rgb buffer failed: %d", ret);
    return;
  }

  // NV12 buffer for output, RGB24 -> NV12
  ret = util::rga::create_rga_buffer(frame_width_, frame_height_, RK_FORMAT_YCbCr_420_SP,
                                     output_nv12_buffer_);
  if (ret != 0) {
    LOGE(TAG, "create rgb buffer failed: %d", ret);
    return;
  }
}

void DetSource::DestroyRgaBuffers() {
  util::rga::release_rga_buffer(rgb_buffer_);
  util::rga::release_rga_buffer(output_nv12_buffer_);

  rga_buffers_.clear();
  last_success_fd_ = -1;
  ready_.store(false);
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

rga_buffer_t* DetSource::Convert2RGB24(rga_buffer_t* src) {
  if (rgb_buffer_.handle == 0) {
    LOGE(TAG, "failed to acquire RGB rga buffer");
    return nullptr;
  }

  auto ret = imcvtcolor(*src, rgb_buffer_, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "convert to rgb888 failed, %s", imStrError(ret));
    return nullptr;
  }

  return &rgb_buffer_;
}

rga_buffer_t* DetSource::Convert2NV12(rga_buffer_t* src) {
  if (output_nv12_buffer_.handle == 0) {
    LOGE(TAG, "failed to acquire NV12 rga buffer");
    return nullptr;
  }

  auto ret = imcvtcolor(*src, output_nv12_buffer_, RK_FORMAT_RGB_888, RK_FORMAT_YCbCr_420_SP);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "convert to NV12 failed, %s", imStrError(ret));
    return nullptr;
  }

  return &output_nv12_buffer_;
}

void DetSource::MainLoop() {
  LOGI(TAG, "DetSource main loop started");

  while (running_.load()) {
    if (!ready_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    auto begin = std::chrono::steady_clock::now();

    auto src_buffer = GetImportedRgaBuffer();
    if (src_buffer == nullptr) {
      LOGE(TAG, "failed to get imported rga buffer");
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // Convert to RGB24
    rga_buffer_t* rgb_buffer = Convert2RGB24(src_buffer);
    if (rgb_buffer == nullptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    // Detect faces
    auto detected_faces = detector_->Detect(rgb_buffer);

    if (!detected_faces.empty()) {
#if PARALLEL_EXECUTION
      // Parallel crop & extract
      ExtractEmbeddings(rgb_buffer, detected_faces);
#else
      // Extract embeddings
      extractor_->Extract(rgb_buffer, detected_faces);
#endif  // PARALLEL_EXECUTION

      // Check similarity with registered faces
      for (auto& det_face : detected_faces) {
        float max_sim = -1.0f;
        std::string matched_name = "";
        int face_id = -1;
        for (const auto& reg_face : registered_faces_) {
          float sim =
              face::dot_product_neon_512(det_face.embedding.data(), reg_face.embedding.data());
          // float sim = compare_faces(det_face.embedding, reg_face.embedding);
          if (sim > max_sim) {
            max_sim = sim;
            matched_name = reg_face.name;
            face_id = reg_face.id;
          }
        }
        // Set the name if similarity is above threshold
        if (max_sim >= kRecogSimilarityThreshold) {
          std::strncpy(det_face.name, matched_name.c_str(), sizeof(det_face.name) - 1);
          det_face.id = face_id;  // To match the OSD text index(NOTICE: osd_idx = face_id - 1)
        } else {
          det_face.id = -1;
          LOGW(TAG, "No matched face found for detected face, max similarity: %.4f", max_sim);
        }
      }

      // Draw OSD
      DrawOsd(rgb_buffer, detected_faces);
    }

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

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    LOGD(TAG, "Processed one frame in %llu ms, detected: %zu faces", elapsed_ms,
         detected_faces.size());

    infer_count++;
    total_infer_time += elapsed_ms;
    LOGD(TAG, "Processed frame: %u, cost total time: %lu ms, average time per frame: %lu ms",
         infer_count, total_infer_time, total_infer_time / infer_count);
  }

  LOGD(TAG, "DetSource main loop exited");
}

void DetSource::CreateOsdTexts() {
  for (size_t i = 0; i < registered_faces_.size(); ++i) {
    auto osd_text = std::make_unique<OsdText>(registered_faces_[i].name);
    osd_texts_.emplace_back(std::move(osd_text));
  }
}

void DetSource::DestroyOsdTexts() {
  for (auto& osd_text : osd_texts_) {
    osd_text.reset();
  }
  osd_texts_.clear();
}

void DetSource::DrawOsd(rga_buffer_t* buffer, const std::vector<face::FaceLocation>& objects) {
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
    ret = imrectangleTask(job, *buffer, rect, kOsdColor, 2);
    if (IM_STATUS_SUCCESS != ret) {
      LOGE(TAG, "apply draw rectangle task failed, %s", imStrError(ret));
      continue;
    }

    // Draw label
    if (obj.id > 0) {
      const auto& osd = osd_texts_[obj.id - 1];
      int osd_y = std::max(0, y - osd->GetHeight() - 2);
      im_rect osd_rect = {x, osd_y, osd->GetWidth(), osd->GetHeight()};
      auto& osd_img = osd->GetRgaBuffer();
      ret = imcheck(osd_img, *buffer, {}, osd_rect);
      if (IM_STATUS_NOERROR != ret) {
        LOGE(TAG, "draw label check failed, %s", imStrError(ret));
        continue;
      }
      ret = improcessTask(job, osd_img, *buffer, {}, {}, osd_rect, {}, nullptr, IM_SYNC);
      if (IM_STATUS_SUCCESS != ret) {
        LOGE(TAG, "apply draw label task failed, %s", imStrError(ret));
        continue;
      }
    }

    // Draw landmarks
    const float s = 6;
    for (int i = 0; i < 5; ++i) {
      int a = (int) (obj.lmk[i * 2] - s / 2.0f);
      int b = (int) (obj.lmk[i * 2 + 1] - s / 2.0f);
      if (a > frame_width_ || b > frame_height_ || a < 0 || b < 0) {
        continue;
      }

      a = (a) & (~1);
      b = (b) & (~1);

      im_rect lmk_rect{a, b, (int) s, (int) s};
      ret = imfillTask(job, *buffer, lmk_rect, kOsdColor);
      if (IM_STATUS_SUCCESS != ret) {
        LOGE(TAG, "apply draw rectangle task failed, %s", imStrError(ret));
        continue;
      }
    }
  }

  ret = imendJob(job);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "imendJob failed, %s", imStrError(ret));
    imcancelJob(job);
  }
}

#if PARALLEL_EXECUTION

void DetSource::CreateCropBuffers() {
  for (int i = 0; i < MAX_PARALLEL_TASKS; ++i) {
    // Allocate buffer for cropped image
    auto buf_size = kExtractorInputSize * kExtractorInputSize * 3;
    auto vir_addr = (char*) calloc(buf_size, sizeof(char));
    if (vir_addr == nullptr) {
      assert(false && "FaceExtractor alloc system buffer failed");
    }
    // Import to RGA buffer
    auto handle = importbuffer_virtualaddr(vir_addr, buf_size);
    if (handle == 0) {
      assert(false && "FaceExtractor importbuffer_virtualaddr failed");
    }
    rga_buffer_t* buffer = new rga_buffer_t;
    *buffer =
        wrapbuffer_handle(handle, kExtractorInputSize, kExtractorInputSize, RK_FORMAT_RGB_888);
    buffer->vir_addr = vir_addr;
    crop_buffers_.push_back(buffer);
  }
}

void DetSource::DestroyCropBuffers() {
  for (auto& buffer : crop_buffers_) {
    if (buffer->handle > 0) {
      releasebuffer_handle(buffer->handle);
      buffer->handle = 0;
    }
    if (buffer->vir_addr) {
      auto buf_size = kExtractorInputSize * kExtractorInputSize * 3;
      dma_buf_free(buf_size, nullptr, buffer->vir_addr);
      buffer->vir_addr = nullptr;
    }
    delete buffer;
  }
  crop_buffers_.clear();
}

int DetSource::ExtractEmbeddings(rga_buffer_t* src, std::vector<face::FaceLocation>& faces) {
  int num_faces = (int) faces.size();
  if (num_faces == 1) {
    // Single face, no need for parallel
    return ExtractOne(extractors_[0], src, crop_buffers_[0], faces[0]);
  }

  // Multiple faces, parallel execution
  std::latch latch(static_cast<std::ptrdiff_t>(faces.size()));
  for (int i = 0; i < num_faces; ++i) {
    thread_pool_->Post([&, i]() {
      auto& face = faces[i];
      auto& box = face.box;

      auto pool_id = extractors_pool_->Acquire();
      auto* crop_buffer = crop_buffers_[pool_id];
      auto* extractor = extractors_[pool_id];

      // Extract one face
      int ret = ExtractOne(extractor, src, crop_buffer, face);
      if (ret != 0) {
        LOGE(TAG, "ExtractOne failed for face idx %d, ret=%d", i, ret);
      }

      extractors_pool_->Release(pool_id);
      latch.count_down();
    });
  }
  latch.wait();

  return 0;
}

int DetSource::ExtractOne(face::FaceExtractorP* extractor, rga_buffer_t* src, rga_buffer_t* dst,
                          face::FaceLocation& face) {
  auto& box = face.box;
  auto crop_rect = face::make_square_crop(box.x, box.y, box.width, box.height, (float) src->width,
                                          (float) src->height, 1.2f);
  // Convert lmk points to cropped patch space
  float lmk[10] = {0.f};
  for (int i = 0; i < 5; ++i) {
    cv::Point2f p_img(face.lmk[i * 2], face.lmk[i * 2 + 1]);
    cv::Point2f p_patch =
        face::landmark_to_patch_112(p_img, crop_rect, (float) kExtractorInputSize);
    lmk[i * 2] = p_patch.x;
    lmk[i * 2 + 1] = p_patch.y;
  }

  // Crop face using RGA
  im_rect src_rect = {(int) crop_rect.x, (int) crop_rect.y, (int) crop_rect.width,
                      (int) crop_rect.height};
  im_rect dst_rect = {0, 0, kExtractorInputSize, kExtractorInputSize};
  auto ret = imcheck(*src, *dst, src_rect, dst_rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "crop imcheck failed: %s", imStrError((IM_STATUS) ret));
    return -1;
  }
  ret = improcess(*src, *dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    LOGE(TAG, "crop improcess failed: %s", imStrError((IM_STATUS) ret));
    return -2;
  }

  // Extract embedding
  return extractor->Extract(dst, lmk, face.embedding);
}

#endif  // PARALLEL_EXECUTION

}  // namespace det