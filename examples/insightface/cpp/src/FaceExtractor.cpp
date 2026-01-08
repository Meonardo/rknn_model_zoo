#include "FaceExtractor.h"

#include "Common.h"

#define TAG "FaceExtractor"

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

#define TEST_CROP_IMAGE_PATH "/sdcard/Download/test_crop.jpg"

#define ENABLE_PROFILE 0

#if ENABLE_PROFILE
static uint32_t infer_count = 0;
static int64_t infer_npu_time = 0;
#endif

namespace face {

// similarity transform destination points
static const std::vector<cv::Point2f> SIMILARITY_TRANSFORM_DEST = {{38.2946, 51.6963},
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
  cv::Matx<double, 2, 2> U_matx(U.at<double>(0, 0), U.at<double>(0, 1), U.at<double>(1, 0),
                                U.at<double>(1, 1));
  cv::Matx<double, 2, 2> Vt_matx(Vt.at<double>(0, 0), Vt.at<double>(0, 1), Vt.at<double>(1, 0),
                                 Vt.at<double>(1, 1));

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

static cv::Mat norm_crop(cv::Mat& img, float kps[10], int size, cv::Mat& dst_mat) {
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
  cv::warpAffine(img, dst_mat, M, cv::Size(size, size), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                 cv::Scalar(0, 0, 0));

  return M;
}

static inline cv::Rect2f make_square_crop(float x, float y, float w, float h, float img_w,
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

static inline cv::Point2f landmark_to_patch_112(const cv::Point2f& p_img,
                                                const cv::Rect2f& crop /* in img space */,
                                                float out_size /*112*/) {
  float sx = out_size / crop.width;
  float sy = out_size / crop.height;
  return cv::Point2f((p_img.x - crop.x) * sx, (p_img.y - crop.y) * sy);
}

FaceExtractor::FaceExtractor(const std::string& model_path, int frame_width, int frame_height)
    : model_path_(model_path),
      frame_width_(frame_width),
      frame_height_(frame_height),
      initialized_(false),
      full_size_crop_(false),
      rknn_ctx_(0),
      model_input_width_(112),
      model_input_height_(112) {
  LOGI(TAG, "Create FaceExtractor with model: %s", model_path_.c_str());
  auto ret = Init();
  assert(ret == 0 && "FaceExtractor Init failed");
  initialized_ = true;
}

FaceExtractor::~FaceExtractor() {
  LOGI(TAG, "Destroying FaceExtractor with model: %s", model_path_.c_str());

  DeInit();

  LOGI(TAG, "FaceExtractor with model: %s destroyed", model_path_.c_str());
}

int FaceExtractor::Init() {
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
  if (io_num_.n_output != 1) {
    LOGE(TAG, "Only support single output for quantized model");
    return -1;
  }
  deqnt_params_.scale = output_attrs_[0].scale;
  deqnt_params_.zero_point = output_attrs_[0].zp;
#endif

  return 0;
}

void FaceExtractor::DeInit() {
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

void FaceExtractor::CreateRgaBuffers() {
  int buf_size = 0;
  int fd = -1;
  char* vir_addr = nullptr;
  rga_buffer_handle_t handle = 0;

  if (full_size_crop_) {
    // Allocate buffer for source image(full size)
    buf_size = frame_width_ * frame_height_ * 3;
    vir_addr = (char*) calloc(buf_size, sizeof(char));
    if (vir_addr == nullptr) {
      assert(false && "FaceExtractor alloc system buffer failed");
    }
    // Import to RGA buffer
    handle = importbuffer_virtualaddr(vir_addr, buf_size);
    if (handle == 0) {
      assert(false && "FaceExtractor importbuffer_virtualaddr failed");
    }
    src_image_rga_buffer_ =
        wrapbuffer_handle(handle, frame_width_, frame_height_, RK_FORMAT_RGB_888);
    src_image_rga_buffer_.vir_addr = vir_addr;
  } else {
    // Allocate buffer for affine image(model input size)
    buf_size = model_input_width_ * model_input_height_ * 3;
    vir_addr = (char*) calloc(buf_size, sizeof(char));
    if (vir_addr == nullptr) {
      assert(false && "FaceExtractor alloc system buffer failed");
    }
    // Import to RGA buffer
    handle = importbuffer_virtualaddr(vir_addr, buf_size);
    if (handle == 0) {
      assert(false && "FaceExtractor importbuffer_virtualaddr failed");
    }
    affine_image_rga_buffer_ =
        wrapbuffer_handle(handle, model_input_width_, model_input_height_, RK_FORMAT_RGB_888);
    affine_image_rga_buffer_.vir_addr = vir_addr;
  }

  // Allocate buffer for cropped image
  buf_size = model_input_width_ * model_input_height_ * 3;
  vir_addr = (char*) calloc(buf_size, sizeof(char));
  if (vir_addr == nullptr) {
    assert(false && "FaceExtractor alloc system buffer failed");
  }
  // Import to RGA buffer
  handle = importbuffer_virtualaddr(vir_addr, buf_size);
  if (handle == 0) {
    assert(false && "FaceExtractor importbuffer_virtualaddr failed");
  }
  crop_image_rga_buffer_ =
      wrapbuffer_handle(handle, model_input_width_, model_input_height_, RK_FORMAT_RGB_888);
  crop_image_rga_buffer_.vir_addr = vir_addr;
  // Create cropped_mat_ wrapper
  cropped_mat_ = cv::Mat(crop_image_rga_buffer_.height, crop_image_rga_buffer_.width, CV_8UC3,
                         crop_image_rga_buffer_.vir_addr, crop_image_rga_buffer_.width * 3);

#if USE_QUANTIZED_MODEL
  // Try to import input tensor memory into RGA
  fd = input_mems_[0]->fd;
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
    auto ret = util::rga::create_rga_buffer(model_input_width_, model_input_height_,
                                            RK_FORMAT_RGB_888, input_tensor_rga_buffer_);
    if (ret != 0) {
      LOGE(TAG, "create scale buffer(for input tensor memory) failed: %d", ret);
      return;
    }
  }
}

void FaceExtractor::DestroyRgaBuffers() {
  // Release the input tensor RGA buffer
  if (input_tensor_rga_buffer_.handle > 0) {
    releasebuffer_handle(input_tensor_rga_buffer_.handle);
    input_tensor_rga_buffer_.handle = -1;
  }
  // Release the source image RGA buffer
  if (src_image_rga_buffer_.handle > 0) {
    releasebuffer_handle(src_image_rga_buffer_.handle);
    src_image_rga_buffer_.handle = -1;
    if (src_image_rga_buffer_.vir_addr != nullptr) {
      free(src_image_rga_buffer_.vir_addr);
      src_image_rga_buffer_.vir_addr = nullptr;
    }
  }
  // Release the affine image RGA buffer
  if (affine_image_rga_buffer_.handle > 0) {
    releasebuffer_handle(affine_image_rga_buffer_.handle);
    affine_image_rga_buffer_.handle = -1;
    if (affine_image_rga_buffer_.vir_addr != nullptr) {
      free(affine_image_rga_buffer_.vir_addr);
      affine_image_rga_buffer_.vir_addr = nullptr;
    }
  }
  // Release the cropped image RGA buffer
  if (crop_image_rga_buffer_.vir_addr != nullptr) {
    releasebuffer_handle(crop_image_rga_buffer_.handle);
    crop_image_rga_buffer_.handle = -1;
    if (crop_image_rga_buffer_.vir_addr != nullptr) {
      free(crop_image_rga_buffer_.vir_addr);
      crop_image_rga_buffer_.vir_addr = nullptr;
    }
  }
}

rga_buffer_t* FaceExtractor::Crop(cv::Mat& src, FaceLocation& face) {
  if (input_tensor_rga_buffer_.handle == 0) {
    LOGE(TAG, "failed to acquire scale rga buffer");
    return nullptr;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  // Align and crop face
  norm_crop(src, face.lmk, model_input_width_, cropped_mat_);
#if 0
  cv::imwrite(TEST_CROP_IMAGE_PATH, cropped_mat_);
#endif
  auto end = std::chrono::high_resolution_clock::now();
  LOGD(TAG, "Crop time: %lld ms",
       std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

  // Copy src to `src_image_rga_buffer_`
  im_rect dst_rect = {0, 0, (int) model_input_width_, (int) model_input_height_};
  auto ret = imcheck(crop_image_rga_buffer_, input_tensor_rga_buffer_, {}, dst_rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "imcheck failed: %s", imStrError((IM_STATUS) ret));
    return nullptr;
  }
  ret = improcess(crop_image_rga_buffer_, input_tensor_rga_buffer_, {}, {}, dst_rect, {}, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    LOGE(TAG, "improcess failed: %s", imStrError((IM_STATUS) ret));
    return nullptr;
  }

  return &input_tensor_rga_buffer_;
}

rga_buffer_t* FaceExtractor::Crop(rga_buffer_t* src, const cv::Rect2f& rect, float* lmk) {
  auto begin = std::chrono::high_resolution_clock::now();

  im_rect src_rect = {(int) rect.x, (int) rect.y, (int) rect.width, (int) rect.height};
  im_rect dst_rect = {0, 0, (int) model_input_width_, (int) model_input_height_};
  auto ret = imcheck(*src, crop_image_rga_buffer_, src_rect, dst_rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "crop imcheck failed: %s", imStrError((IM_STATUS) ret));
    return nullptr;
  }
  ret = improcess(*src, crop_image_rga_buffer_, {}, src_rect, dst_rect, {}, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    LOGE(TAG, "crop improcess failed: %s", imStrError((IM_STATUS) ret));
    return nullptr;
  }

  auto end = std::chrono::high_resolution_clock::now();
  LOGD(TAG, "Crop time: %lld ms",
       std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

  return &crop_image_rga_buffer_;
}

rga_buffer_t* FaceExtractor::Affine(cv::Mat& src, float *lmk) {
  if (input_tensor_rga_buffer_.handle == 0) {
    LOGE(TAG, "failed to acquire scale rga buffer");
    return nullptr;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  // Align and crop face
  norm_crop(src, lmk, model_input_width_, cropped_mat_);
#if 0
  cv::imwrite(TEST_CROP_IMAGE_PATH, cropped_mat_);
#endif
  auto end = std::chrono::high_resolution_clock::now();
  LOGD(TAG, "Affine time: %lld ms",
       std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

  // Copy src to `src_image_rga_buffer_`
  im_rect dst_rect = {0, 0, (int) model_input_width_, (int) model_input_height_};
  auto ret = imcheck(crop_image_rga_buffer_, input_tensor_rga_buffer_, {}, dst_rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "imcheck failed: %s", imStrError((IM_STATUS) ret));
    return nullptr;
  }
  ret = improcess(crop_image_rga_buffer_, input_tensor_rga_buffer_, {}, {}, dst_rect, {}, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    LOGE(TAG, "improcess failed: %s", imStrError((IM_STATUS) ret));
    return nullptr;
  }

  return &input_tensor_rga_buffer_;
}

int FaceExtractor::Extract(rga_buffer_t* input_buffer, std::vector<FaceLocation>& faces) {
  if (!initialized_) {
    LOGE(TAG, "FaceExtractor not initialized");
    return -1;
  }

  if (full_size_crop_) {  // Use full size crop
    return ExtractFromFullSize(input_buffer, faces);
  }

  // Crop from boxes then resize to model input size
  for (auto& face : faces) {
    auto& boxes = face.box;
    // Convert crop area from source image
    auto crop_rect =
        make_square_crop(boxes.x, boxes.y, boxes.width, boxes.height, (float) input_buffer->width,
                         (float) input_buffer->height, 1.2f);
    // Convert lmk points to cropped patch space
    float lmk[10] = {0.f};
    for (int i = 0; i < 5; ++i) {
      cv::Point2f p_img(face.lmk[i * 2], face.lmk[i * 2 + 1]);
      cv::Point2f p_patch = landmark_to_patch_112(p_img, crop_rect, (float) model_input_width_);
      lmk[i * 2] = p_patch.x;
      lmk[i * 2 + 1] = p_patch.y;
    }

    auto* cropped_buffer = Crop(input_buffer, crop_rect, lmk);
    if (cropped_buffer == nullptr) {
      continue;
    }

    // Wrap cropped_buffer to cv::Mat for affine transform
    cv::Mat crop_rgb(cropped_buffer->height, cropped_buffer->width, CV_8UC3,
                     cropped_buffer->vir_addr, cropped_buffer->width * 3);

    auto* affine_buffer = Affine(crop_rgb, lmk);
    if (affine_buffer == nullptr) {
      continue;
    }

    // Inference
    rknn_tensor_mem* mem = input_mems_[0];
    if (!mem) {
      LOGE(TAG, "input mem is null");
      return -4;
    }

#if USE_QUANTIZED_MODEL
    if (input_tensor_rga_buffer_.handle <= 0) {  // Use DMA buffer, this should not happen
      memcpy(mem->virt_addr, affine_buffer->vir_addr, mem->size);
    }
#else
    // Convert RGB24 to NHWC float32
    util::rknn::rgb24_to_nhwc_float_stride((uint8_t*) affine_buffer->vir_addr, model_input_width_,
                                           model_input_height_, (int) affine_buffer->wstride,
                                           (float*) mem->virt_addr, 1.0f);
#endif

    // Inference
    auto ret = rknn_run(rknn_ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_run fail ret=%d", ret);
      return -5;
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

    // Post-process: get embeddings
    int embeddings_size = output_attrs_[0].n_elems;
#if USE_QUANTIZED_MODEL
    int8_t* output_data = (int8_t*) output_mems_[0]->virt_addr;
    // Dequantize output
    face.embedding.resize(embeddings_size);
    for (int i = 0; i < embeddings_size; ++i) {
      float e = util::rknn::deqnt_affine_to_f32(output_data[i], deqnt_params_.scale,
                                                deqnt_params_.zero_point);
      face.embedding[i] = e;
    }
#else
    // Directly read float32 output
    float* output_data = (float*) output_mems_[0]->virt_addr;
    face.embedding = std::vector<float>(output_data, output_data + embeddings_size);
#endif
  }

  return 0;
}

int FaceExtractor::ExtractFromFullSize(rga_buffer_t* input_buffer,
                                       std::vector<FaceLocation>& faces) {
  auto begin = std::chrono::high_resolution_clock::now();

  // Copy src to `src_image_rga_buffer_`
  im_rect dst_rect = {0, 0, input_buffer->width, input_buffer->height};
  auto ret = imcheck(*input_buffer, src_image_rga_buffer_, {}, dst_rect);
  if (IM_STATUS_NOERROR != ret) {
    LOGE(TAG, "imcheck failed: %s", imStrError((IM_STATUS) ret));
    return -2;
  }
  ret = improcess(*input_buffer, src_image_rga_buffer_, {}, {}, dst_rect, {}, IM_SYNC);
  if (ret != IM_STATUS_SUCCESS) {
    LOGE(TAG, "improcess failed: %s", imStrError((IM_STATUS) ret));
    return -3;
  }
  auto end_copy = std::chrono::high_resolution_clock::now();
  LOGD(TAG, "Copy input image time: %lld ms",
       std::chrono::duration_cast<std::chrono::milliseconds>(end_copy - begin).count());

  cv::Mat src_rgb(frame_height_, frame_width_, CV_8UC3, src_image_rga_buffer_.vir_addr,
                  src_image_rga_buffer_.width * 3);

  for (auto& face : faces) {
    auto* cropped_buffer = Crop(src_rgb, face);
    if (cropped_buffer == nullptr) {
      continue;
    }

    // Copy input data to input tensor memory
    rknn_tensor_mem* mem = input_mems_[0];
    if (!mem) {
      LOGE(TAG, "input mem is null");
      return -4;
    }

#if USE_QUANTIZED_MODEL
    if (input_tensor_rga_buffer_.handle <= 0) {  // Use DMA buffer, this should not happen
      memcpy(mem->virt_addr, cropped_buffer->vir_addr, mem->size);
    }
#else
    // Convert RGB24 to NHWC float32
    util::rknn::rgb24_to_nhwc_float_stride((uint8_t*) cropped_buffer->vir_addr, model_input_width_,
                                           model_input_height_, (int) cropped_buffer->wstride,
                                           (float*) mem->virt_addr, 1.0f);
#endif

    auto pre_process_end = std::chrono::high_resolution_clock::now();
    LOGD(TAG, "Pre-process time: %lld ms",
         std::chrono::duration_cast<std::chrono::milliseconds>(pre_process_end - begin).count());

    // Inference
    auto ret = rknn_run(rknn_ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      LOGE(TAG, "rknn_run fail ret=%d", ret);
      return -5;
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

    // Post-process: get embeddings
    int embeddings_size = output_attrs_[0].n_elems;
#if USE_QUANTIZED_MODEL
    int8_t* output_data = (int8_t*) output_mems_[0]->virt_addr;
    // Dequantize output
    face.embedding.resize(embeddings_size);
    for (int i = 0; i < embeddings_size; ++i) {
      float e = util::rknn::deqnt_affine_to_f32(output_data[i], deqnt_params_.scale,
                                                deqnt_params_.zero_point);
      face.embedding[i] = e;
    }
#else
    // Directly read float32 output
    float* output_data = (float*) output_mems_[0]->virt_addr;
    face.embedding = std::vector<float>(output_data, output_data + embeddings_size);
#endif
  }

  return 0;
}

}  // namespace face