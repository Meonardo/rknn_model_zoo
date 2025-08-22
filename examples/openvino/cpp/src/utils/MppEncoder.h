//
// Created by Meonardo on 7/7/2024.
//

#ifndef GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_UTILS_MPPENCODER_H_
#define GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_UTILS_MPPENCODER_H_

#include <thread>
#include <atomic>
#include <cstring>
#include <memory>

#include "mpp_frame.h"
#include "rk_mpi.h"

#include "im2d.h"

#include "../VideoSink.h"

// forward declaration
class RgaBufferPool;

namespace v_enc {

typedef void (*MppEncoderFrameCallback)(void *userdata, const char *data, size_t size);

// Encoder parameters
struct MppEncoderParams {
  RK_U32 width;
  RK_U32 height;
  RK_U32 hor_stride;
  RK_U32 ver_stride;
  MppFrameFormat fmt;
  MppCodingType type;

  RK_U32 osd_enable;
  RK_U32 osd_mode;
  RK_U32 split_mode;
  RK_U32 split_arg;
  RK_U32 split_out;

  RK_U32 user_data_enable;
  RK_U32 roi_enable;

  // rate control runtime parameter
  RK_S32 fps_in_flex;
  RK_S32 fps_in_den;
  RK_S32 fps_in_num;
  RK_S32 fps_out_flex;
  RK_S32 fps_out_den;
  RK_S32 fps_out_num;
  RK_S32 bps;
  RK_S32 bps_max;
  RK_S32 bps_min;
  RK_S32 rc_mode;
  RK_S32 gop_mode;
  RK_S32 gop_len;
  RK_S32 vi_len;

  // general qp control
  RK_S32 qp_init;
  RK_S32 qp_max;
  RK_S32 qp_max_i;
  RK_S32 qp_min;
  RK_S32 qp_min_i;
  RK_S32 qp_max_step; /* delta qp between each two P frame */
  RK_S32 qp_delta_ip; /* delta qp between I and P */
  RK_S32 qp_delta_vi; /* delta qp between vi and P */

  RK_U32 constraint_set;
  RK_U32 rotation;
  RK_U32 mirroring;
  RK_U32 flip;

  MppEncHeaderMode header_mode;
  MppEncSeiMode sei_mode;
};

class MppEncoder {
 public:
  MppEncoder(EncodedVideoInfo encoded_video_info, void *userdata);
  ~MppEncoder();

  void SetCallback(MppEncoderFrameCallback cb) {
	this->callback_ = cb;
  }

  // Async encode, encoded data will be returned by callback
  MPP_RET Encode(const rga_buffer_t *rga_buffer);
  MPP_RET Encode(const VideoFrameSlot &frame);
  // Sync encode, return value is the size of encoded data
  size_t Encode(const rga_buffer_t *rga_buffer, char **encoded_data);
  size_t Encode(const VideoFrameSlot &frame, char **encoded_data);
  bool Stop();

  int Reset();
  [[nodiscard]] size_t GetFrameSize() const;

 private:
  EncodedVideoInfo encoded_video_info_;
  void *userdata_;
  MppCtx mpp_ctx_;
  MppApi *mpp_mpi_;
  MppBufferGroup buf_grp_;
  MppBuffer frm_buf_;

  MppEncoderFrameCallback callback_;

  // global flow control flag
  RK_U32 frm_eos_;
  RK_U32 pkt_eos_;
  RK_S32 frame_num_;

  // encoder config set
  MppEncCfg cfg_;
  std::unique_ptr<std::thread> input_thread_;
  std::unique_ptr<std::thread> output_thread_;
  std::atomic<bool> loop_end_;

  // resources
  size_t header_size_;
  size_t frame_size_;

  RK_S64 first_frame_;
  RK_S64 first_pkt_;
  RK_S64 last_pkt_;

  MppEncoderParams enc_params_;

  int Init(MppEncoderParams &params);
  int InitParams(MppEncoderParams &params);
  int SetupEncCfg();
  void CleanEncoderData();

  // void InputThread();
  void OutputThread();
};

} // namespace v_enc

#endif //GSTREAMERANDROID_MEDIACORE_SRC_MAIN_CPP_UTILS_MPPENCODER_H_
