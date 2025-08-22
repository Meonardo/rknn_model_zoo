//
// Created by Meonardo on 7/7/2024.
//
#include "MppEncoder.h"

#include <cstdio>
#include <utility>

#include "mpp_buffer.h"
#include "mpp_dmabuf.h"
#include "mpp_time.h"

#include "RgaBufferPool.h"
#include "../Common.h"

#define MPP_ALIGN(x, a)         (((x)+(a)-1)&~((a)-1))
#define SZ_4K 4096
#define TAG "MppEncoder"

namespace v_enc {
constexpr int kBaseVideoBps = 4000000;

int MppEncoder::InitParams(MppEncoderParams &params) {
  memcpy(&enc_params_, &params, sizeof(MppEncoderParams));

  // encode parameter from default
  if (enc_params_.hor_stride == 0) {
	enc_params_.hor_stride = MPP_ALIGN(enc_params_.width, 16);
  }
  if (enc_params_.ver_stride == 0) {
	enc_params_.ver_stride = MPP_ALIGN(enc_params_.height, 16);
  }
  if (enc_params_.fps_in_den == 0) {
	enc_params_.fps_in_den = 1;
  }
  if (enc_params_.fps_in_num == 0) {
	enc_params_.fps_in_num = kDefaultFps;
  }
  if (enc_params_.fps_out_den == 0) {
	enc_params_.fps_out_den = 1;
  }
  if (enc_params_.fps_out_num == 0) {
	enc_params_.fps_out_num = kDefaultFps;
  }
  if (enc_params_.gop_len == 0) {
	enc_params_.gop_len = enc_params_.fps_out_num;
  }
  if (!enc_params_.bps) {
	enc_params_.bps = kBaseVideoBps;
  }

  // update resource parameter
  switch (enc_params_.fmt & MPP_FRAME_FMT_MASK) {
	case MPP_FMT_YUV420SP:
	case MPP_FMT_YUV420P: {
	  frame_size_ =
		  MPP_ALIGN(enc_params_.hor_stride, 64) * MPP_ALIGN(enc_params_.ver_stride, 64) * 3 / 2;
	}
	  break;

	case MPP_FMT_YUV422_YUYV :
	case MPP_FMT_YUV422_YVYU :
	case MPP_FMT_YUV422_UYVY :
	case MPP_FMT_YUV422_VYUY :
	case MPP_FMT_YUV422P :
	case MPP_FMT_YUV422SP : {
	  frame_size_ =
		  MPP_ALIGN(enc_params_.hor_stride, 64) * MPP_ALIGN(enc_params_.ver_stride, 64) * 2;
	}
	  break;
	case MPP_FMT_RGB444 :
	case MPP_FMT_BGR444 :
	case MPP_FMT_RGB555 :
	case MPP_FMT_BGR555 :
	case MPP_FMT_RGB565 :
	case MPP_FMT_BGR565 :
	case MPP_FMT_RGB888 :
	case MPP_FMT_BGR888 :
	case MPP_FMT_RGB101010 :
	case MPP_FMT_BGR101010 :
	case MPP_FMT_ARGB8888 :
	case MPP_FMT_ABGR8888 :
	case MPP_FMT_BGRA8888 :
	case MPP_FMT_RGBA8888 : {
	  frame_size_ =
		  MPP_ALIGN(enc_params_.hor_stride, 64) * MPP_ALIGN(enc_params_.ver_stride, 64);
	}
	  break;

	default: {
	  frame_size_ =
		  MPP_ALIGN(enc_params_.hor_stride, 64) * MPP_ALIGN(enc_params_.ver_stride, 64) * 4;
	}
	  break;
  }

  if (MPP_FRAME_FMT_IS_FBC(enc_params_.fmt)) {
	if ((enc_params_.fmt & MPP_FRAME_FBC_MASK) == MPP_FRAME_FBC_AFBC_V1)
	  header_size_ =
		  MPP_ALIGN(MPP_ALIGN(enc_params_.width, 16) * MPP_ALIGN(enc_params_.height, 16) / 16,
					SZ_4K);
	else
	  header_size_ = MPP_ALIGN(enc_params_.width, 16) * MPP_ALIGN(enc_params_.height, 16) / 16;
  } else {
	header_size_ = 0;
  }

  return 0;
}

int MppEncoder::SetupEncCfg() {
  MPP_RET ret;

  ret = mpp_enc_cfg_init(&cfg_);
  if (ret) {
	LOGE(TAG, "mpp_enc_cfg_init failed ret %d", ret);
	return -1;
  }

  int low_delay = 1;
  mpp_enc_cfg_set_s32(cfg_, "base:low_delay", low_delay);

  mpp_enc_cfg_set_s32(cfg_, "prep:width", (RK_S32)enc_params_.width);
  mpp_enc_cfg_set_s32(cfg_, "prep:height", (RK_S32)enc_params_.height);
  mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", (RK_S32)enc_params_.hor_stride);
  mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", (RK_S32)enc_params_.ver_stride);
  mpp_enc_cfg_set_s32(cfg_, "prep:format", enc_params_.fmt);

  mpp_enc_cfg_set_s32(cfg_, "rc:mode", enc_params_.rc_mode);

  /* fix input / output frame rate */
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_flex", enc_params_.fps_in_flex);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_num", enc_params_.fps_in_num);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_denom", enc_params_.fps_in_den);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_flex", enc_params_.fps_out_flex);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_num", enc_params_.fps_out_num);
  mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_denom", enc_params_.fps_out_den);
  mpp_enc_cfg_set_s32(cfg_, "rc:gop", enc_params_.gop_len);
  if (low_delay) {
	mpp_enc_cfg_set_s32(cfg_, "rc:max_reenc_times", 0);
  }

  /* drop frame or not when bitrate overflow */
  mpp_enc_cfg_set_u32(cfg_, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_NORMAL);
  mpp_enc_cfg_set_u32(cfg_, "rc:drop_thd", 20);        /* 20% of max bps */
  mpp_enc_cfg_set_u32(cfg_, "rc:drop_gap", 1);         /* Do not continuous drop frame */

  /* setup bitrate for different rc_mode */
  mpp_enc_cfg_set_s32(cfg_, "rc:bps_target", enc_params_.bps);
  switch (enc_params_.rc_mode) {
	case MPP_ENC_RC_MODE_FIXQP : {
	  /* do not setup bitrate on FIXQP mode */
	}
	  break;
	case MPP_ENC_RC_MODE_CBR : {
	  /* CBR mode has narrow bound */
	  mpp_enc_cfg_set_s32(cfg_,
						  "rc:bps_max",
						  enc_params_.bps_max ? enc_params_.bps_max : enc_params_.bps * 17 / 16);
	  mpp_enc_cfg_set_s32(cfg_,
						  "rc:bps_min",
						  enc_params_.bps_min ? enc_params_.bps_min : enc_params_.bps * 15 / 16);
	}
	  break;
	case MPP_ENC_RC_MODE_VBR :
	case MPP_ENC_RC_MODE_AVBR : {
	  /* VBR mode has wide bound */
	  mpp_enc_cfg_set_s32(cfg_,
						  "rc:bps_max",
						  enc_params_.bps_max ? enc_params_.bps_max : enc_params_.bps * 17 / 16);
	  mpp_enc_cfg_set_s32(cfg_,
						  "rc:bps_min",
						  enc_params_.bps_min ? enc_params_.bps_min : enc_params_.bps * 1 / 16);
	}
	  break;
	default : {
	  /* default use CBR mode */
	  mpp_enc_cfg_set_s32(cfg_,
						  "rc:bps_max",
						  enc_params_.bps_max ? enc_params_.bps_max : enc_params_.bps * 17 / 16);
	  mpp_enc_cfg_set_s32(cfg_,
						  "rc:bps_min",
						  enc_params_.bps_min ? enc_params_.bps_min : enc_params_.bps * 15 / 16);
	}
	  break;
  }

  /* setup qp for different codec and rc_mode */
  switch (enc_params_.type) {
	case MPP_VIDEO_CodingAVC :
	case MPP_VIDEO_CodingHEVC : {
	  switch (enc_params_.rc_mode) {
		case MPP_ENC_RC_MODE_FIXQP : {
		  RK_S32 fix_qp = enc_params_.qp_init;

		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_init", fix_qp);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max", fix_qp);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min", fix_qp);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max_i", fix_qp);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min_i", fix_qp);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_ip", 0);
		}
		  break;
		case MPP_ENC_RC_MODE_CBR :
		case MPP_ENC_RC_MODE_VBR :
		case MPP_ENC_RC_MODE_AVBR : {
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_init", -1);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max", 51);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min", 10);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max_i", 51);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min_i", 10);
		  mpp_enc_cfg_set_s32(cfg_, "rc:qp_ip", 2);
		  mpp_enc_cfg_set_s32(cfg_, "rc:fqp_min_i", 10);
		  mpp_enc_cfg_set_s32(cfg_, "rc:fqp_max_i", 51);
		  mpp_enc_cfg_set_s32(cfg_, "rc:fqp_min_p", 10);
		  mpp_enc_cfg_set_s32(cfg_, "rc:fqp_max_p", 51);
		}
		  break;
		default : {
		  LOGE(TAG, "unsupported encoder rc mode %d", enc_params_.rc_mode);
		}
		  break;
	  }
	}
	  break;
	case MPP_VIDEO_CodingVP8 : {
	  /* vp8 only setup base qp range */
	  mpp_enc_cfg_set_s32(cfg_, "rc:qp_init", 40);
	  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max", 127);
	  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min", 0);
	  mpp_enc_cfg_set_s32(cfg_, "rc:qp_max_i", 127);
	  mpp_enc_cfg_set_s32(cfg_, "rc:qp_min_i", 0);
	  mpp_enc_cfg_set_s32(cfg_, "rc:qp_ip", 6);
	}
	  break;
	case MPP_VIDEO_CodingMJPEG : {
	  /* jpeg use special codec config to control qtable */
	  mpp_enc_cfg_set_s32(cfg_, "jpeg:q_factor", 80);
	  mpp_enc_cfg_set_s32(cfg_, "jpeg:qf_max", 99);
	  mpp_enc_cfg_set_s32(cfg_, "jpeg:qf_min", 1);
	}
	  break;
	default : {
	}
	  break;
  }

  /* setup codec  */
  mpp_enc_cfg_set_s32(cfg_, "codec:type", enc_params_.type);
  switch (enc_params_.type) {
	case MPP_VIDEO_CodingAVC : {
	  /*//!< AVC Profile IDC definitions
	  typedef enum h264e_profile_t {
		H264_PROFILE_FREXT_CAVLC444     = 44,   //!< YUV 4:4:4/14 "CAVLC 4:4:4"
		H264_PROFILE_BASELINE           = 66,   //!< YUV 4:2:0/8  "Baseline"
		H264_PROFILE_MAIN               = 77,   //!< YUV 4:2:0/8  "Main"
		H264_PROFILE_EXTENDED           = 88,   //!< YUV 4:2:0/8  "Extended"
		H264_PROFILE_HIGH               = 100,  //!< YUV 4:2:0/8  "High"
		H264_PROFILE_HIGH10             = 110,  //!< YUV 4:2:0/10 "High 10"
		H264_PROFILE_HIGH422            = 122,  //!< YUV 4:2:2/10 "High 4:2:2"
		H264_PROFILE_HIGH444            = 244,  //!< YUV 4:4:4/14 "High 4:4:4"
		H264_PROFILE_MVC_HIGH           = 118,  //!< YUV 4:2:0/8  "Multiview High"
		H264_PROFILE_STEREO_HIGH        = 128   //!< YUV 4:2:0/8  "Stereo High"
	  } H264Profile;*/
	  int profile = 100;
	  mpp_enc_cfg_set_s32(cfg_, "h264:profile", profile);
	  /*//!< AVC Level IDC definitions
	  typedef enum {
		H264_LEVEL_1_0                  = 10,   //!< qcif@15fps
		H264_LEVEL_1_b                  = 99,   //!< qcif@15fps
		H264_LEVEL_1_1                  = 11,   //!< cif@7.5fps
		H264_LEVEL_1_2                  = 12,   //!< cif@15fps
		H264_LEVEL_1_3                  = 13,   //!< cif@30fps
		H264_LEVEL_2_0                  = 20,   //!< cif@30fps
		H264_LEVEL_2_1                  = 21,   //!< half-D1@@25fps
		H264_LEVEL_2_2                  = 22,   //!< D1@12.5fps
		H264_LEVEL_3_0                  = 30,   //!< D1@25fps
		H264_LEVEL_3_1                  = 31,   //!< 720p@30fps
		H264_LEVEL_3_2                  = 32,   //!< 720p@60fps
		H264_LEVEL_4_0                  = 40,   //!< 1080p@30fps
		H264_LEVEL_4_1                  = 41,   //!< 1080p@30fps
		H264_LEVEL_4_2                  = 42,   //!< 1080p@60fps
		H264_LEVEL_5_0                  = 50,   //!< 3K@30fps
		H264_LEVEL_5_1                  = 51,   //!< 4K@30fps
		H264_LEVEL_5_2                  = 52,   //!< 4K@60fps
		H264_LEVEL_6_0                  = 60,   //!< 8K@30fps
		H264_LEVEL_6_1                  = 61,   //!< 8K@60fps
		H264_LEVEL_6_2                  = 62,   //!< 8K@120fps
	  } H264Level;*/
	  mpp_enc_cfg_set_s32(cfg_, "h264:level", 42);

	  if (profile >= 100) { // disabled when use base/main profile
		mpp_enc_cfg_set_s32(cfg_, "h264:cabac_en", 1);
		mpp_enc_cfg_set_s32(cfg_, "h264:cabac_idc", 0);
		mpp_enc_cfg_set_s32(cfg_, "h264:trans8x8", 1);
	  }

	  mpp_enc_cfg_set_s32(cfg_, "h264:stream_type", 0); // Annex B, start code: 00 00 00 01
	}
	  break;
	case MPP_VIDEO_CodingHEVC : {
	  /*typedef enum {
		H265_LEVEL_NONE = 0,
		H265_LEVEL1 = 30,
		H265_LEVEL2 = 60,
		H265_LEVEL2_1 = 63,
		H265_LEVEL3 = 90,
		H265_LEVEL3_1 = 93,
		H265_LEVEL4 = 120,
		H265_LEVEL4_1 = 123,
		H265_LEVEL5 = 150,
		H265_LEVEL5_1 = 153,
		H265_LEVEL5_2 = 156,
		H265_LEVEL6 = 180,
		H265_LEVEL6_1 = 183,
		H265_LEVEL6_2 = 186,
		H265_LEVEL8_5 = 255,
	  } H265Level;*/
	  mpp_enc_cfg_set_s32(cfg_, "h265:profile", 1);
	  mpp_enc_cfg_set_s32(cfg_, "h265:level", 120); // 1080@30fps
	  break;
	}
	case MPP_VIDEO_CodingMJPEG :
	case MPP_VIDEO_CodingVP8 : {
	}
	  break;
	default : {
	  LOGE(TAG, "unsupported encoder coding type %d", enc_params_.type);
	}
	  break;
  }

  if (enc_params_.split_mode) {
	LOGD(TAG, "%p split mode %d arg %d out %d", mpp_ctx_,
		 enc_params_.split_mode, enc_params_.split_arg, enc_params_.split_out);
	mpp_enc_cfg_set_s32(cfg_, "split:mode", (RK_S32)enc_params_.split_mode);
	mpp_enc_cfg_set_s32(cfg_, "split:arg", 8);
	mpp_enc_cfg_set_s32(cfg_, "split:out", MPP_ENC_SPLIT_OUT_LOWDELAY);
  }

  ret = mpp_mpi_->control(mpp_ctx_, MPP_ENC_SET_CFG, cfg_);
  if (ret) {
	LOGE(TAG, "mpi control enc set cfg_ failed ret %d", ret);
	return ret;
  }

  if (enc_params_.type == MPP_VIDEO_CodingAVC || enc_params_.type == MPP_VIDEO_CodingHEVC) {
	enc_params_.header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
	ret = mpp_mpi_->control(mpp_ctx_, MPP_ENC_SET_HEADER_MODE, &enc_params_.header_mode);
	if (ret) {
	  LOGE(TAG, "mpi control enc set header mode failed ret %d", ret);
	  return ret;
	}
  }

  return ret;
}

MppEncoder::MppEncoder(EncodedVideoInfo encoded_video_info, void *userdata) :
	encoded_video_info_(std::move(encoded_video_info)),
	userdata_(userdata),
	mpp_ctx_(nullptr),
	mpp_mpi_(nullptr),
	buf_grp_(nullptr),
	frm_buf_(nullptr),
	callback_(nullptr),
	frm_eos_(0),
	pkt_eos_(0),
	frame_num_(0),
	cfg_(nullptr),
	input_thread_(nullptr),
	output_thread_(nullptr),
	loop_end_(false),
	header_size_(0),
	frame_size_(0),
	first_frame_(0),
	first_pkt_(0),
	last_pkt_(0), enc_params_{} {
  enc_params_.width = encoded_video_info_.width;
  enc_params_.height = encoded_video_info_.height;
  if (encoded_video_info_.codec == 0) {
	enc_params_.type = MPP_VIDEO_CodingAVC;
  } else if (encoded_video_info_.codec == 1) {
	enc_params_.type = MPP_VIDEO_CodingHEVC;
  } else if (encoded_video_info_.codec == 2) {
	enc_params_.type = MPP_VIDEO_CodingMJPEG;
  } else {
	LOGE(TAG, "unsupported codec type %d", encoded_video_info_.codec);
	assert(false);
  }
  enc_params_.bps = (RK_S32)encoded_video_info_.bitrate;
  enc_params_.fps_out_num = encoded_video_info.fps;
  enc_params_.rc_mode = MppEncRcMode(encoded_video_info_.rate_control);
  if (encoded_video_info.pixel_format == 0) {
	enc_params_.fmt = MPP_FMT_YUV420SP;
  } else if (encoded_video_info.pixel_format == 1) {
	enc_params_.fmt = MPP_FMT_RGBA8888;
  } else {
	LOGE(TAG, "unsupported pixel format %d", encoded_video_info.pixel_format);
	assert(false);
  }

  // init encoder
  Init(enc_params_);
}

MppEncoder::~MppEncoder() {
  // stop threads
  Stop();
  // clean encoder data
  CleanEncoderData();
}

int MppEncoder::Init(MppEncoderParams &params) {
  // init encoder parameters
  InitParams(params);

  int ret = mpp_create(&mpp_ctx_, &mpp_mpi_);
  if (ret) {
	LOGE(TAG, "mpp_create failed ret %d", ret);
	CleanEncoderData();
	return ret;
  }

  LOGD(TAG,
	   "mpp encoder started: w %d h %d type %d",
	   enc_params_.width,
	   enc_params_.height,
	   enc_params_.type);

  MppPollType timeout = MPP_POLL_BLOCK;
  ret = mpp_mpi_->control(mpp_ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout);
  if (MPP_OK != ret) {
	LOGE(TAG, "mpi control set output timeout %d ret %d", timeout, ret);
	CleanEncoderData();
	return ret;
  }

  ret = mpp_init(mpp_ctx_, MPP_CTX_ENC, enc_params_.type);
  if (ret) {
	LOGE(TAG, "mpp_init failed ret %d", ret);
	CleanEncoderData();
	return ret;
  }

  // configure encoder
  SetupEncCfg();

  return ret;
}

void MppEncoder::CleanEncoderData() {
  if (mpp_ctx_) {
	mpp_destroy(mpp_ctx_);
	mpp_ctx_ = nullptr;
  }

  if (cfg_) {
	mpp_enc_cfg_deinit(cfg_);
	cfg_ = nullptr;
  }

  if (frm_buf_ != nullptr) {
	mpp_buffer_put(frm_buf_);
	frm_buf_ = nullptr;
  }

  if (buf_grp_ != nullptr) {
	mpp_buffer_group_put(buf_grp_);
	buf_grp_ = nullptr;
  }
}

bool MppEncoder::Stop() {
  if (loop_end_.load()) {
	LOGW(TAG, "encoder is already stopped");
	return false;
  }

  // send eos
  frm_eos_ = 1;
  pkt_eos_ = 1;

  VideoFrameSlot eos_frame = {0};
  eos_frame.eos = true;
  eos_frame.fd = -1;
  eos_frame.width = enc_params_.width;
  eos_frame.height = enc_params_.height;
  eos_frame.width_stride = enc_params_.hor_stride;
  eos_frame.height_stride = enc_params_.ver_stride;
  Encode(eos_frame);

  loop_end_.store(true);

  if (input_thread_ != nullptr && input_thread_->joinable()) {
	input_thread_->join();
	input_thread_.reset();
  }
  if (output_thread_ != nullptr && output_thread_->joinable()) {
	output_thread_->join();
	output_thread_.reset();
  }

  return true;
}

int MppEncoder::Reset() {
  if (mpp_mpi_ != nullptr) {
	mpp_mpi_->reset(mpp_ctx_);
  }
  return 0;
}

size_t MppEncoder::GetFrameSize() const {
  return frame_size_;
}

MPP_RET MppEncoder::Encode(const rga_buffer_t *rga_buffer) {
  if (rga_buffer->fd > 0) {
	VideoFrameSlot f = {0};
	f.fd = rga_buffer->fd;
	f.width = rga_buffer->width;
	f.height = rga_buffer->height;
	f.width_stride = rga_buffer->wstride;
	f.height_stride = rga_buffer->hstride;
	if (rga_buffer->width != enc_params_.width || rga_buffer->height != enc_params_.height) {
	  LOGE(TAG, "input frame size %dx%d is not equal to encoder size %dx%d",
		   rga_buffer->width, rga_buffer->height, enc_params_.width, enc_params_.height);
	  return MPP_NOK;
	}

	return Encode(f);
  } else {
	if (output_thread_ == nullptr) {
	  loop_end_.store(false);
	  output_thread_ = std::make_unique<std::thread>(&MppEncoder::OutputThread, this);
	}

	MPP_RET ret = MPP_OK;
	auto frm_size = rga_buffer->wstride * rga_buffer->hstride * 3 / 2;
	if (frm_buf_ == nullptr) {
	  ret = mpp_buffer_group_get_internal(&buf_grp_,
										  MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CACHABLE);
	  if (ret) {
		LOGE(TAG, "failed to get mpp buffer group ret %d", ret);
		return ret;
	  }

	  ret = mpp_buffer_get(buf_grp_, &frm_buf_, frm_size);
	  if (ret) {
		LOGE(TAG, "failed to get buffer for input frame ret %d", ret);
		return ret;
	  }
	}

	MppFrame frame = nullptr;

	void *buf = mpp_buffer_get_ptr(frm_buf_);
	if (buf == nullptr) {
	  LOGE(TAG, "failed to get buffer ptr");
	  return MPP_NOK;
	}

	mpp_buffer_sync_begin(frm_buf_);

	// copy the rga vir_address to buffer
	memcpy(buf, rga_buffer->vir_addr, frm_size);

	mpp_buffer_sync_end(frm_buf_);

	ret = mpp_frame_init(&frame);
	if (ret) {
	  LOGE(TAG, "mpp_frame_init failed ret %d", ret);
	  return ret;
	}

	mpp_frame_set_width(frame, rga_buffer->width);
	mpp_frame_set_height(frame, rga_buffer->height);
	mpp_frame_set_hor_stride(frame, rga_buffer->wstride);
	mpp_frame_set_ver_stride(frame, rga_buffer->hstride);
	mpp_frame_set_fmt(frame, enc_params_.fmt);
	mpp_frame_set_eos(frame, frm_eos_);
	mpp_frame_set_buffer(frame, frm_buf_);
	mpp_frame_set_buf_size(frame, frm_size);

	if (!first_frame_) {
	  first_frame_ = mpp_time();
	}

	do {
	  ret = mpp_mpi_->encode_put_frame(mpp_ctx_, frame);
	  if (ret) {
		msleep(1);
		LOGW(TAG, "encode_put_frame failed ret %d", ret);
	  }
	} while (ret);

	// release resources
	if (frm_buf_ != nullptr) {
	  mpp_buffer_put(frm_buf_);
	}
	mpp_frame_deinit(&frame);

	if (frm_eos_) {
	  LOGW(TAG, "found eos frame");
	  return ret;
	}

	return ret;
  }

}

MPP_RET MppEncoder::Encode(const VideoFrameSlot &f) {
  if (output_thread_ == nullptr) { // for testing
	loop_end_.store(false);
	output_thread_ = std::make_unique<std::thread>(&MppEncoder::OutputThread, this);
  }

  MPP_RET ret = MPP_OK;
  MppFrame frame = nullptr;
  MppBuffer buffer = nullptr;

  if (f.fd > 0) {
	MppBufferInfo info;
	memset(&info, 0, sizeof(MppBufferInfo));
	info.type = MPP_BUFFER_TYPE_EXT_DMA;
	info.fd = f.fd;
	info.size = f.width_stride * f.height_stride * 3 / 2;
	info.index = f.fd;
	ret = mpp_buffer_import(&buffer, &info);
	if (ret) {
	  LOGE(TAG, "mpp_buffer_import failed ret %d", ret);
	  return ret;
	}
	mpp_buffer_sync_end(buffer);
  }

  ret = mpp_frame_init(&frame);
  if (ret) {
	LOGE(TAG, "mpp_frame_init failed ret %d", ret);
	return ret;
  }

  mpp_frame_set_width(frame, f.width);
  mpp_frame_set_height(frame, f.height);
  mpp_frame_set_hor_stride(frame, f.width_stride);
  mpp_frame_set_ver_stride(frame, f.height_stride);
  mpp_frame_set_fmt(frame, enc_params_.fmt);
  mpp_frame_set_eos(frame, frm_eos_);
  mpp_frame_set_buffer(frame, buffer);

  if (!first_frame_) {
	first_frame_ = mpp_time();
  }

  do {
	ret = mpp_mpi_->encode_put_frame(mpp_ctx_, frame);
	if (ret) {
	  msleep(1);
	  LOGW(TAG, "encode_put_frame failed ret %d", ret);
	}
  } while (ret);

  // release resources
  if (buffer != nullptr) {
	mpp_buffer_put(buffer);
  }
  mpp_frame_deinit(&frame);

  if (frm_eos_) {
	LOGW(TAG, "found eos frame");
	return ret;
  }

  return ret;
}

size_t MppEncoder::Encode(const rga_buffer_t* rga_buffer, char** encoded_data) {
  if (rga_buffer->fd <= 0) {
	LOGE(TAG, "invalid input frame fd %d", rga_buffer->fd);
	return 0;
  }
  VideoFrameSlot f = {0};
  f.fd = rga_buffer->fd;
  f.width = rga_buffer->width;
  f.height = rga_buffer->height;
  f.width_stride = rga_buffer->wstride;
  f.height_stride = rga_buffer->hstride;
  if (rga_buffer->width != enc_params_.width || rga_buffer->height != enc_params_.height) {
	LOGE(TAG, "input frame size %dx%d is not equal to encoder size %dx%d",
		 rga_buffer->width, rga_buffer->height, enc_params_.width, enc_params_.height);
	return MPP_NOK;
  }

  return Encode(f, encoded_data);
}

size_t MppEncoder::Encode(const VideoFrameSlot &f, char** encoded_data) {
  MPP_RET ret = MPP_OK;
  MppFrame frame = nullptr;
  MppBuffer buffer = nullptr;
  MppPacket packet = nullptr;

  if (f.fd > 0) {
	MppBufferInfo info;
	memset(&info, 0, sizeof(MppBufferInfo));
	info.type = MPP_BUFFER_TYPE_EXT_DMA;
	info.fd = f.fd;
	if (encoded_video_info_.pixel_format == 0) {
	  // YUV420SP
	  info.size = f.width_stride * f.height_stride * 3 / 2;
	} else {
	  // MPP_FMT_RGBA8888
	  info.size = f.width_stride * f.height_stride * 4;
	}
	info.index = f.fd;
	ret = mpp_buffer_import(&buffer, &info);
	if (ret) {
	  LOGE(TAG, "mpp_buffer_import failed ret %d", ret);
	  return -1;
	}
  }

  ret = mpp_frame_init(&frame);
  if (ret) {
	LOGE(TAG, "mpp_frame_init failed ret %d", ret);
	return -2;
  }

  mpp_frame_set_width(frame, f.width);
  mpp_frame_set_height(frame, f.height);
  mpp_frame_set_hor_stride(frame, f.width_stride);
  mpp_frame_set_ver_stride(frame, f.height_stride);
  mpp_frame_set_fmt(frame, enc_params_.fmt);
  mpp_frame_set_eos(frame, frm_eos_);
  mpp_frame_set_buffer(frame, buffer);

  // put frame into encoder
  do {
	ret = mpp_mpi_->encode_put_frame(mpp_ctx_, frame);
	if (ret) {
	  msleep(1);
	  LOGW(TAG, "encode_put_frame failed ret %d", ret);
	}
  } while (ret);

  if (buffer != nullptr) {
	mpp_buffer_put(buffer);
  }

  // get encoded packet
  ret = mpp_mpi_->encode_get_packet(mpp_ctx_, &packet);
  if (ret || packet == nullptr) {
	LOGW(TAG, "encode_get_packet failed ret %d packet %p", ret, packet);
	return -3;
  }

  auto ptr = mpp_packet_get_pos(packet);
  auto len = mpp_packet_get_length(packet);

  // assign the encoded data
  *encoded_data = (char*)ptr;

  // release resources
  mpp_frame_deinit(&frame);
  mpp_packet_deinit(&packet);

  // return the length of encoded data
  return len;
}

void MppEncoder::OutputThread() {
  LOGI(TAG, "Output thread started");

  MPP_RET ret = MPP_OK;
  MppPacket packet = nullptr;
  void *ptr;
  size_t len;
  char log_buf[256];
  RK_S32 log_size = sizeof(log_buf) - 1;
  RK_S32 log_len = 0;

  while (!loop_end_.load()) {
	ret = mpp_mpi_->encode_get_packet(mpp_ctx_, &packet);
	if (ret || packet == nullptr) {
	  msleep(1);
	  LOGW(TAG, "encode_get_packet failed ret %d packet %p", ret, packet);
	  continue;
	}

	last_pkt_ = mpp_time();

	ptr = mpp_packet_get_pos(packet);
	len = mpp_packet_get_length(packet);
	log_size = sizeof(log_buf) - 1;
	log_len = 0;

	if (!first_pkt_) {
	  first_pkt_ = mpp_time();
	}

	pkt_eos_ = mpp_packet_get_eos(packet);

	// callback here !!!
	if (callback_ != nullptr && len > 0) {
	  callback_(userdata_, (const char *)ptr, len);
	}

	/*log_len += snprintf(log_buf + log_len, log_size - log_len,
						"encoded frame %-4d", frame_count_out_);

	LOGD(TAG, "chn %d %s", chn, log_buf);*/

	mpp_packet_deinit(&packet);

	if (frm_eos_) {
	  pkt_eos_ = 1;
	  LOGD(TAG, "found eos frame");
	  break;
	}

	if (pkt_eos_) {
	  LOGD(TAG, "found last packet");
	  mpp_assert(frm_eos_);
	  break;
	}
  }

  LOGI(TAG, "Output thread stopped");
}

} // namespace v_enc