//
// Created by Meonardo on 7/25/2024.
//

#include "CameraSource.h"

#include <fcntl.h>
#include <linux/v4l2-subdev.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include <chrono>
#include <thread>

#include "Common.h"

#define TAG "CameraSource"

#define FMT_NUM_PLANES 1

#define ENABLE_TEST_FILE_OUTPUT 0

constexpr auto kCameraCapturePixelFormat = V4L2_PIX_FMT_MPEG;
constexpr auto kCameraCaptureWidth = kBaseVideoWidth;
constexpr auto kCameraCaptureHeight = kBaseVideoHeight;
constexpr auto kCameraRequestBufferCount = kMaxDecodedFrameBufferCount;
constexpr auto kDefaultRetrieveTimeInMs = 1500;
constexpr auto kWaitTimeInMs = 30;

#if ENABLE_TEST_FILE_OUTPUT
static FILE* test_file = nullptr;
static void write_jpeg_to_file(const char* data, size_t size) {
  if (!test_file) {
    test_file = fopen("/data/local/tmp/camera_capture.jpg", "wb");
    if (!test_file) {
      LOGE(TAG, "Failed to open test file for writing");
      return;
    }
  }

  fwrite(data, 1, size, test_file);
  fflush(test_file);
}
#endif

// Wrap ioctl() to spin on EINTR
static int camera_source_ioctl(int fd, int req, void* arg) {
  int ret = -99;

  while ((ret = ioctl(fd, req, arg))) {
    if (ret == -1 && (EINTR != errno && EAGAIN != errno)) {
      LOGW(TAG, "ioctl failed: req:%d, ret = %d, errno %d", req, ret, errno);
      break;
    }

    // sleep 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return ret;
}

CameraSource::CameraSource(std::string_view id, std::string_view device_id)
    : id_(id),
      device_id_(device_id),
      device_fd_(-1),
      buffer_type_(V4L2_BUF_TYPE_VIDEO_CAPTURE),
      worker_(nullptr),
      capturing_(false),
      running_(false),
      mpp_decoder_(
          new v_dec::MppDecoder(MPP_VIDEO_CodingMJPEG, kBaseVideoWidth, kBaseVideoHeight, this)),
      max_width_(kCameraCaptureWidth),
      max_height_(kCameraCaptureHeight),
      nv12_buffer_pool_(nullptr),
      current_frame_({}) {
  LOGI(TAG, "CameraSource %s created, device id=%s", id_.c_str(), device_id_.c_str());

  mpp_decoder_->SetCallback([](void* userdata, RK_U32 width_stride, RK_U32 height_stride,
                               RK_U32 width, RK_U32 height, MppFrameFormat format, int fd) {
    auto* source = reinterpret_cast<CameraSource*>(userdata);
    source->OnDecodedFrame({width, height, width_stride, height_stride, format, fd});
  });
  worker_ = std::make_unique<std::thread>(&CameraSource::Start, this);
}

CameraSource::~CameraSource() {
  LOGI(TAG, "Call destroying CameraSource %s", id_.c_str());
  Stop();
  // Release decoder
  if (mpp_decoder_) {
    mpp_decoder_->PutPacket(nullptr, 0, 1);  // send EOS
    // sleep for a while to wait for the decoder to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    delete mpp_decoder_;
    mpp_decoder_ = nullptr;
  }
  LOGI(TAG, "CameraSource %s destroyed", id_.c_str());
}

void CameraSource::OpenCamera() {
  // open camera device
  device_fd_ = open(device_id_.c_str(), O_RDWR, 0);
  if (device_fd_ < 0) {
    LOGE(TAG, "open camera %s failed, reason: %s", id_.c_str(), strerror(errno));
    return;
  }
  LOGI(TAG, "camera[%d] %s opened", device_fd_, id_.c_str());

  // query device capabilities
  v4l2_capability caps = {0};
  if (camera_source_ioctl(device_fd_, VIDIOC_QUERYCAP, &caps) != 0) {
    LOGE(TAG, "query camera %s capabilities failed", id_.c_str());
    CloseCamera();
    return;
  }
  if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
      !(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
    LOGE(TAG, "capture buffer format not supported");
    CloseCamera();
    return;
  }
  if (!(caps.capabilities & V4L2_CAP_STREAMING)) {
    LOGE(TAG, "streaming IO not Supported");
    CloseCamera();
    return;
  }

  v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (caps.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    LOGI(TAG, "camera %s supports multi-plane, use it", id_.c_str());
  }
  buffer_type_ = (v4l2_buf_type) fmt.type;

  {
    v4l2_fmtdesc fmt_desc = {0};

    fmt_desc.index = 0;
    fmt_desc.type = fmt.type;
    while (!camera_source_ioctl(device_fd_, VIDIOC_ENUM_FMT, &fmt_desc)) {
      LOGI(TAG, "pixel format: [%c%c%c%c], description: [%s]", fmt_desc.pixelformat & 0xFF,
           (fmt_desc.pixelformat >> 8) & 0xFF, (fmt_desc.pixelformat >> 16) & 0xFF,
           (fmt_desc.pixelformat >> 24) & 0xFF, fmt_desc.description);
      fmt_desc.index++;
    }
  }

  {
    v4l2_frmsizeenum frame_size = {0};
    frame_size.index = 0;
    frame_size.pixel_format = kCameraCapturePixelFormat;
    while (ioctl(device_fd_, VIDIOC_ENUM_FRAMESIZES, &frame_size) == 0) {
      if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
        LOGI(TAG, "resolution: %ux%u", frame_size.discrete.width, frame_size.discrete.height);
      } else if (frame_size.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
        LOGI(TAG, "min resolution: %ux%u", frame_size.stepwise.min_width,
             frame_size.stepwise.min_height);
        LOGI(TAG, "max resolution: %ux%u", frame_size.stepwise.max_width,
             frame_size.stepwise.max_height);

        // save the max resolution
        max_width_ = frame_size.stepwise.max_width;
        max_height_ = frame_size.stepwise.max_height;

        if (frame_size.stepwise.max_width != kCameraCaptureWidth ||
            frame_size.stepwise.max_height != kCameraCaptureHeight) {
          LOGI(TAG, "resize needed, original [%dx%d] to [%dx%d]", frame_size.stepwise.max_width,
               frame_size.stepwise.max_height, kCameraCaptureWidth, kCameraCaptureHeight);
          max_width_ = frame_size.stepwise.max_width;
          max_height_ = frame_size.stepwise.max_height;

          nv12_buffer_pool_ =
              std::make_unique<RgaBufferPool>(kMaxDecodedFrameBufferCount, RK_FORMAT_YCbCr_420_SP);
        }
        LOGI(TAG, "step Resolution: %ux%u", frame_size.stepwise.step_width,
             frame_size.stepwise.step_height);
      }
      frame_size.index++;
    }
  }

  // set format
  fmt.fmt.pix.pixelformat = kCameraCapturePixelFormat;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  fmt.fmt.pix.width = max_width_;
  fmt.fmt.pix.height = max_height_;

  if (camera_source_ioctl(device_fd_, VIDIOC_S_FMT, &fmt) == -1) {
    LOGE(TAG, "set camera %s format failed", id_.c_str());
    CloseCamera();
    return;
  }

  if (camera_source_ioctl(device_fd_, VIDIOC_G_FMT, &fmt) == -1) {
    LOGE(TAG, "get camera %s format failed", id_.c_str());
    CloseCamera();
    return;
  }

  LOGI(TAG, "Camera(%s) size: [%dx%d], pixel format: [%c%c%c%c]", id_.c_str(), fmt.fmt.pix.width,
       fmt.fmt.pix.height, fmt.fmt.pix.pixelformat & 0xFF, (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
       (fmt.fmt.pix.pixelformat >> 16) & 0xFF, (fmt.fmt.pix.pixelformat >> 24) & 0xFF);

  // request buffers
  v4l2_requestbuffers req = {0};
  req.count = kCameraRequestBufferCount;
  req.type = fmt.type;
  req.memory = V4L2_MEMORY_MMAP;
  if (camera_source_ioctl(device_fd_, VIDIOC_REQBUFS, &req) == -1) {
    LOGE(TAG, "request camera %s buffers failed", id_.c_str());
    CloseCamera();
    return;
  }
  if (req.count != kCameraRequestBufferCount) {
    LOGE(TAG, "request camera %s buffers failed, count=%d", id_.c_str(), req.count);
    CloseCamera();
    return;
  }

  // map buffers
  size_t buf_len = 0;
  for (uint32_t i = 0; i < req.count; i++) {
    v4l2_buffer buf = {0};
    buf.type = fmt.type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    v4l2_plane planes[FMT_NUM_PLANES] = {0};
    if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == fmt.type) {
      buf.m.planes = planes;
      buf.length = FMT_NUM_PLANES;
    }

    if (camera_source_ioctl(device_fd_, VIDIOC_QUERYBUF, &buf) == -1) {
      LOGE(TAG, "query camera %s buffer %d failed", id_.c_str(), i);
      CloseCamera();
      return;
    }

    CameraFrame frame = {nullptr};
    if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf.type) {
      buf_len = buf.m.planes[0].length;
      frame.start = mmap(nullptr, buf_len, PROT_READ | PROT_WRITE, MAP_SHARED, device_fd_,
                         buf.m.planes[0].m.mem_offset);
    } else {
      buf_len = buf.length;
      frame.start =
          mmap(nullptr, buf_len, PROT_READ | PROT_WRITE, MAP_SHARED, device_fd_, buf.m.offset);
    }
    if (frame.start == MAP_FAILED) {
      LOGE(TAG, "map camera %s buffer %d failed", id_.c_str(), i);
      CloseCamera();
      return;
    }

    // record buffer length for unmap
    frame.length = buf_len;

    v4l2_exportbuffer export_buffer = {0};
    export_buffer.type = buf.type;
    export_buffer.index = i;
    export_buffer.flags = O_CLOEXEC;
    if (camera_source_ioctl(device_fd_, VIDIOC_EXPBUF, &export_buffer) < 0) {
      LOGE(TAG, "export camera %s buffer %d failed", id_.c_str(), i);
      CloseCamera();
      return;
    } else {
      // LOGD(TAG, "export camera %s buffer %d success, fd=%d", id_.c_str(), i, export_buffer.fd);
      frame.export_fd = export_buffer.fd;
      if (max_width_ != kCameraCaptureWidth || max_height_ != kCameraCaptureHeight) {
        // create rga buffer for resizing
        auto buffer = std::make_unique<rga_buffer_t>();
        if (ImportRgaBuffer(frame.export_fd, buffer.get())) {
          rga_buffers_[frame.export_fd] = std::move(buffer);
        }
      }
    }
    frames_.push_back(frame);
  }

  for (int i = 0; i < kCameraRequestBufferCount; i++) {
    v4l2_plane planes[FMT_NUM_PLANES] = {0};

    v4l2_buffer buffer = {0};
    buffer.type = fmt.type;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = i;

    if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == fmt.type) {
      buffer.m.planes = planes;
      buffer.length = FMT_NUM_PLANES;
    }

    if (camera_source_ioctl(device_fd_, VIDIOC_QBUF, &buffer) == -1) {
      LOGE(TAG, "queue camera %s buffer %d failed", id_.c_str(), i);
      CloseCamera();
      return;
    }
  }

  // start capturing
  if (camera_source_ioctl(device_fd_, VIDIOC_STREAMON, &fmt.type) == -1) {
    LOGE(TAG, "start camera %s streaming failed", id_.c_str());
    CloseCamera();
    return;
  }

  // check the video device is ready to capture
  if (!CheckCameraAvailable()) {
    CloseCamera();
    return;
  }

  CaptureLoop();
}

void CameraSource::CloseCamera() {
  v4l2_buffer buf = {0};

  if (device_fd_ < 0) {
    return;
  }

  // stop capturing
  camera_source_ioctl(device_fd_, VIDIOC_STREAMOFF, &buffer_type_);

  // unmap buffers
  for (auto& frame : frames_) {
    if (frame.start != nullptr && frame.length > 0) {
      munmap(frame.start, frame.length);
      close(frame.export_fd);
    }
  }
  frames_.clear();

  // release rga buffers
  if (!rga_buffers_.empty()) {
    for (auto& [key, buffer] : rga_buffers_) {
      releasebuffer_handle(buffer->handle);
    }
    rga_buffers_.clear();
  }
  if (nv12_buffer_pool_) {
    nv12_buffer_pool_.reset();
  }

  // close camera device
  LOGI(TAG, "close camera[%d] %s", device_fd_, id_.c_str());
  close(device_fd_);
  device_fd_ = -1;
}

void CameraSource::Start() {
  running_.store(true);
  while (running_.load()) {
    OpenCamera();

    if (!running_.load()) {
      // we are stopping, break the loop
      // no need to wait for restart
      break;
    }
    // sleep 3s before restart
    std::this_thread::sleep_for(std::chrono::seconds(3));
    LOGW(TAG, "camera %s is restarting in 3seconds...", id_.c_str());
  }
}

void CameraSource::Stop() {
  running_.store(false);
  capturing_.store(false);
  // stop capture thread
  if (worker_ && worker_->joinable()) {
    worker_->join();
    worker_.reset();
  }
  // close camera
  CloseCamera();
}

void CameraSource::CaptureLoop() {
  // skip some frames at start
  for (int i = 0; i < kCameraRequestBufferCount; i++) {
    int index = RetrieveFrame();
    if (index >= 0) {
      PutFrame(index);
    }
  }

  LOGI(TAG, "camera[%d] %s opened, start capture loop", device_fd_, id_.c_str());
  capturing_.store(true);
  // flag to refresh frames for each sink
  bool need_refresh_frames = true;
  // start capture loop
  while (capturing_.load()) {
    // record current time
    auto start = std::chrono::steady_clock::now();

    // check if the camera is available
    if (!CheckCameraAvailable()) {
      capturing_.store(false);
      CloseCamera();

      {
        // notify sinks this is the end of the stream, may restart the camera later
        std::lock_guard<std::mutex> lock(sink_mutex_);
        for (auto sink : sinks_) {
          current_frame_ = {kCameraCaptureWidth,
                            kCameraCaptureHeight,
                            kCameraCaptureWidth,
                            kCameraCaptureHeight,
                            MPP_FMT_YUV420SP,
                            0,
                            false,
                            true};
          sink->OnVideoFrame(current_frame_);
        }
      }

      break;
    }

    // get the frame from camera
    int index = RetrieveFrame();
    if (index < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kDefaultWaitingTimeInMs));
      continue;
    }

    auto& frame = frames_[index];

// int fd = frame.export_fd;
#if ENABLE_TEST_FILE_OUTPUT
    write_jpeg_to_file((const char*) frame.start, frame.length);
#endif
    // decode the frame
    mpp_decoder_->PutPacket(frame.start, frame.length, 0);

    // put the frame back to camera
    PutFrame(index);

    // record current time
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // keep the frame rate with `kDefaultFps`
    if (diff.count() < kWaitTimeInMs) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kWaitTimeInMs - diff.count()));
    }
  }
}

int CameraSource::RetrieveFrame() {
  v4l2_buffer buf = {0};
  buf.type = buffer_type_;
  buf.memory = V4L2_MEMORY_MMAP;

  v4l2_plane planes[FMT_NUM_PLANES] = {0};
  if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buffer_type_) {
    buf.m.planes = planes;
    buf.length = FMT_NUM_PLANES;
  }

  if (-1 == camera_source_ioctl(device_fd_, VIDIOC_DQBUF, &buf)) {
    LOGE(TAG, "dequeue camera %s buffer failed", id_.c_str());
    return -1;
  }

  if (buf.index > kCameraRequestBufferCount) {
    LOGE(TAG, "buffer index out of bounds");
    return -1;
  }

  if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buffer_type_) {
    buf.bytesused = buf.m.planes[0].bytesused;
  }

  return (int) buf.index;
}

int CameraSource::PutFrame(int index) {
  if (index < 0) {
    return 0;
  }

  v4l2_buffer buf = {0};
  buf.type = buffer_type_;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = index;

  v4l2_plane planes[FMT_NUM_PLANES];
  if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buffer_type_) {
    buf.m.planes = planes;
    buf.length = FMT_NUM_PLANES;
  }

  // tell kernel it's ok to overwrite this frame
  if (-1 == camera_source_ioctl(device_fd_, VIDIOC_QBUF, &buf)) {
    LOGE(TAG, "enqueue camera %s buffer failed", id_.c_str());
  }

  return 0;
}

void CameraSource::AddVideoSink(VideoSink* sink) {
  LOGI(TAG, "CameraSource %s add VideoSink", id_.c_str());

  std::lock_guard<std::mutex> lock(sink_mutex_);

  if (std::find(sinks_.begin(), sinks_.end(), sink) == sinks_.end()) {
    sinks_.push_back(sink);
  } else {
    LOGW(TAG, "CameraSource %s VideoSink already added", id_.c_str());
  }
}

void CameraSource::RemoveVideoSink(VideoSink* sink) {
  LOGI(TAG, "CameraSource %s remove VideoSink", id_.c_str());

  std::lock_guard<std::mutex> lock(sink_mutex_);

  auto it = std::find(sinks_.begin(), sinks_.end(), sink);
  if (it != sinks_.end()) {
    sinks_.erase(it);
  } else {
    LOGW(TAG, "CameraSource %s VideoSink not found", id_.c_str());
  }
}

VideoFrameSlot& CameraSource::GetVideoFrame() {
  return current_frame_;
}

bool CameraSource::CheckCameraAvailable() {
  if (device_fd_ < 0) {
    LOGE(TAG, "camera %s not available", id_.c_str());
    return false;
  }

  struct pollfd poll_fd = {0};
  poll_fd.fd = device_fd_;
  poll_fd.events = POLLIN;
  if (poll(&poll_fd, 1, kDefaultRetrieveTimeInMs) <= 0) {
    LOGE(TAG, "camera[%d] %s not ready for capturing", device_fd_, id_.c_str());
    return false;
  }

  return true;
}

void CameraSource::OnDecodedFrame(const VideoFrameSlot& frame) {
  if (frame.fd <= 0) {
    LOGE(TAG, "invalid fd");
    return;
  }

  std::lock_guard<std::mutex> lock(sink_mutex_);
  for (auto sink : sinks_) {
    sink->OnVideoFrame(frame);
  }
}

bool CameraSource::ImportRgaBuffer(int fd, rga_buffer_t* buffer) const {
  im_handle_param_t handle_param;
  handle_param.width = max_width_;
  handle_param.height = max_height_;
  handle_param.format = RK_FORMAT_YCbCr_420_SP;  // NV12

  // get rga buffer handle from decoded buffer
  rga_buffer_handle_t handle = importbuffer_fd(fd, &handle_param);
  if (handle <= 0) {
    LOGE(TAG, "camera source rga import dma buffer failed, fd=%d", fd);
    return false;
  }

  *buffer =
      wrapbuffer_handle(handle, (int) max_width_, (int) max_height_, (int) handle_param.format,
                        (int) MPP_ALIGN(max_width_, 16), (int) MPP_ALIGN(max_height_, 16));
  buffer->fd = fd;

  return true;
}

bool CameraSource::Resize(rga_buffer_t* src, rga_buffer_t* dst) {
  im_rect rect = {0, 0, (int) kCameraCaptureWidth, (int) kCameraCaptureHeight};
  auto ret = imcheck(*src, *dst, {}, rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "resize check failed, %s", imStrError(ret));
    return false;
  }

  ret = improcess(*src, *dst, {}, {}, rect, {}, IM_SYNC);
  if (IM_STATUS_SUCCESS != ret) {
    LOGE(TAG, "scale failed, %s", imStrError(ret));
    return false;
  }

  return true;
}