//
// Created by Meonardo on 7/7/2024.
//

#include "MppDecoder.h"

#include <memory>

#include "../Common.h"
#include "mpp_common.h"
#include "mpp_env.h"
#include "mpp_mem.h"
#include "mpp_time.h"

#define TAG "MppDecoder"

#define DEFAULT_PACKET_SIZE SZ_4K

namespace v_dec {
typedef struct DecBufMgrImpl_t {
  MppDecBufMode buf_mode;
  RK_U32 buf_count;
  RK_U32 buf_size;
  MppBufferGroup group;
  MppBuffer* bufs;
} DecBufMgrImpl;

static MPP_RET dec_buf_mgr_init(DecBufMgr* mgr) {
  DecBufMgrImpl* impl = nullptr;
  MPP_RET ret = MPP_NOK;

  if (mgr) {
    impl = mpp_calloc(DecBufMgrImpl, 1);
    if (impl) {
      ret = MPP_OK;
    } else {
      LOGE(TAG, "failed to create decoder buffer manager");
    }

    *mgr = impl;
  }

  return ret;
}

static void dec_buf_mgr_deinit(DecBufMgr mgr) {
  auto impl = (DecBufMgrImpl*) mgr;

  if (nullptr == impl) return;

  /* release buffer group for half internal and external mode */
  if (impl->group) {
    mpp_buffer_group_put(impl->group);
    impl->group = nullptr;
  }

  /* release the buffers used in external mode */
  if (impl->buf_count && impl->bufs) {
    RK_U32 i;

    for (i = 0; i < impl->buf_count; i++) {
      if (impl->bufs[i]) {
        mpp_buffer_put(impl->bufs[i]);
        impl->bufs[i] = nullptr;
      }
    }

    MPP_FREE(impl->bufs);
  }

  MPP_FREE(impl);
}

static MppBufferGroup dec_buf_mgr_setup(DecBufMgr mgr, RK_U32 size, RK_U32 count,
                                        MppDecBufMode mode) {
  auto* impl = (DecBufMgrImpl*) mgr;
  MPP_RET ret = MPP_NOK;

  if (!impl) return nullptr;

  /* cleanup old buffers if previous buffer group exists */
  if (impl->group) {
    if (mode != impl->buf_mode) {
      /* switch to different buffer mode just release old buffer group */
      mpp_buffer_group_put(impl->group);
      impl->group = nullptr;
    } else {
      /* otherwise just cleanup old buffers */
      mpp_buffer_group_clear(impl->group);
    }

    /* if there are external mode old buffers do cleanup */
    if (impl->bufs) {
      RK_U32 i;

      for (i = 0; i < impl->buf_count; i++) {
        if (impl->bufs[i]) {
          mpp_buffer_put(impl->bufs[i]);
          impl->bufs[i] = nullptr;
        }
      }

      MPP_FREE(impl->bufs);
    }
  }

  switch (mode) {
    case MPP_DEC_BUF_HALF_INT: {
      /* reuse previous half internal buffer group and just reconfig limit */
      if (nullptr == impl->group) {
        ret = mpp_buffer_group_get_internal(&impl->group, MPP_BUFFER_TYPE_ION);
        if (ret) {
          LOGE(TAG, "get mpp internal buffer group failed ret %d\n", ret);
          break;
        }
      }
      /* Use limit config to limit buffer count and buffer size */
      ret = mpp_buffer_group_limit_config(impl->group, size, count);
      if (ret) {
        LOGE(TAG, "limit buffer group failed ret %d\n", ret);
      }
    } break;
    case MPP_DEC_BUF_INTERNAL: {
      /* do nothing just keep buffer group empty */
      assert(nullptr == impl->group);
      ret = MPP_OK;
    } break;
    case MPP_DEC_BUF_EXTERNAL: {
      RK_U32 i;
      MppBufferInfo commit;

      impl->bufs = mpp_calloc(MppBuffer, count);
      if (!impl->bufs) {
        LOGE(TAG, "create %d external buffer record failed\n", count);
        break;
      }

      /* reuse previous external buffer group */
      if (nullptr == impl->group) {
        ret = mpp_buffer_group_get_external(&impl->group, MPP_BUFFER_TYPE_ION);
        if (ret) {
          LOGE(TAG, "get mpp external buffer group failed ret %d\n", ret);
          break;
        }
      }

      /*
       * NOTE: Use default misc allocater here as external allocator for demo.
       * But in practical case the external buffer could be GraphicBuffer or gst dmabuf.
       * The misc allocator will cause the print at the end like:
       * ~MppBufferService cleaning misc group
       */
      commit.type = MPP_BUFFER_TYPE_ION;
      commit.size = size;

      for (i = 0; i < count; i++) {
        ret = mpp_buffer_get(nullptr, &impl->bufs[i], size);
        if (ret || nullptr == impl->bufs[i]) {
          LOGE(TAG, "get misc buffer failed ret %d\n", ret);
          break;
        }

        commit.index = i;
        commit.ptr = mpp_buffer_get_ptr(impl->bufs[i]);
        commit.fd = mpp_buffer_get_fd(impl->bufs[i]);

        ret = mpp_buffer_commit(impl->group, &commit);
        if (ret) {
          LOGE(TAG, "external buffer commit failed ret %d", ret);
          break;
        }
      }
    } break;
    default: {
      LOGE(TAG, "unsupported buffer mode %d", mode);
    } break;
  }

  if (ret) {
    dec_buf_mgr_deinit(impl);
    impl = nullptr;
  } else {
    impl->buf_count = count;
    impl->buf_size = size;
    impl->buf_mode = mode;
  }

  return impl ? impl->group : nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

MppDecoder::MppDecoder(MppCodingType type, RK_U32 width, RK_U32 height, void* user_data)
    : use_stream_buffer_(false),
      type_(type),
      format_(MPP_FMT_YUV420SP),
      width_(width),
      height_(height),
      buf_mode_(MPP_DEC_BUF_EXTERNAL),
      ctx_(nullptr),
      mpi_(nullptr),
      loop_end_(false),
      buf_mgr_(nullptr),
      frm_grp_(nullptr),
      packet_(nullptr),
      packet_size_(DEFAULT_PACKET_SIZE),
      frame_(nullptr),
      frame_count_(0),
      buffer_pool_(nullptr),
      encoded_buffer_(nullptr),
      dec_input_thread_(nullptr),
      dec_output_thread_(nullptr),
      callback_(nullptr),
      user_data_(user_data) {
  InitDecoderData();
}

MppDecoder::~MppDecoder() {
  DeInitDecoderData();
}

void MppDecoder::InitDecoderData() {
  // init buffer group
  auto ret = dec_buf_mgr_init(&buf_mgr_);
  if (ret) {
    LOGE(TAG, "dec_buf_mgr_init failed");
    return;
  }

  ret = mpp_packet_init(&packet_, nullptr, 0);
  if (ret) {
    LOGE(TAG, "mpp_packet_init failed");
    return;
  }

  // create mpi
  ret = mpp_create(&ctx_, &mpi_);
  if (ret) {
    LOGE(TAG, "mpp_create failed");
    return;
  }

  // configs before init decoder
  {
    RK_U32 split_mode = 0;
    ret = mpi_->control(ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &split_mode);
    if (ret) {
      LOGE(TAG, "Failed to set split mode %d ret %d", split_mode, ret);
      return;
    }
  }

  // init decoder
  ret = mpp_init(ctx_, MPP_CTX_DEC, type_);
  if (ret) {
    LOGE(TAG, "mpp_init failed");
    return;
  }

  // NOTE: timeout value please refer to MppPollType definition
  //  0   - non-block call (default)
  // -1   - block call
  // +val - timeout value in ms
  {
    MppPollType timeout = MPP_POLL_BLOCK;
    MppParam param = &timeout;

    ret = mpi_->control(ctx_, MPP_SET_OUTPUT_TIMEOUT, param);
    if (ret) {
      LOGE(TAG, "Failed to set output timeout %d ret %d", timeout, ret);
      return;
    }
  }

  // set fast mode for decoder
  {
    bool enabled = true;
    ret = mpi_->control(ctx_, MPP_DEC_SET_IMMEDIATE_OUT, &enabled);
    if (ret) {
      LOGE(TAG, "Failed to set fast decode %d ret %d", enabled, ret);
      return;
    }

    /*ret = mpi_->control(ctx_, MPP_DEC_SET_DISABLE_ERROR, &enabled);
    if (ret) {
      LOGE(TAG, "Failed to set disable error ret %d", ret);
      return;
    }*/
  }

  if (type_ == MPP_VIDEO_CodingAVC) {
    // set buffer group
    auto buffer_size = MPP_ALIGN(width_, 16) * MPP_ALIGN(height_, 16) * 2;
    CommitBufferGroup(width_, height_, buffer_size);
  }

  // create decoding threads
  if (use_stream_buffer_) {
    dec_input_thread_ = std::make_unique<std::thread>(&MppDecoder::Feed, this);
    // set thread name
    pthread_setname_np(dec_input_thread_->native_handle(), "MppDecodeInputThread");
  }

  dec_output_thread_ = std::make_unique<std::thread>(&MppDecoder::Decode, this);
  // set thread name
  pthread_setname_np(dec_output_thread_->native_handle(), "MppDecodingThread");
}

bool MppDecoder::CommitBufferGroup(RK_U32 w, RK_U32 h, size_t size) {
  if (w == 0 || h == 0) {
    return false;  // no resolution info, this will be called in decode thread
  }
  frm_grp_ = dec_buf_mgr_setup(buf_mgr_, size, kMaxDecodedFrameBufferCount, buf_mode_);
  /* Set buffer to mpp decoder */
  auto ret = mpi_->control(ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
  if (ret != 0) {
    LOGE(TAG, "failed to setup buffer group for decoder");
    frm_grp_ = nullptr;
    return false;
  }

  LOGI(TAG, "commit buffer group success");

  return true;
}

void MppDecoder::DeInitDecoderData() {
  // stop the decoding thread
  loop_end_.store(true);

  if (use_stream_buffer_) {
    if (dec_input_thread_ && dec_input_thread_->joinable()) {
      dec_input_thread_->join();
      dec_input_thread_.reset();
    }
  }
  if (dec_output_thread_ && dec_output_thread_->joinable()) {
    dec_output_thread_->join();
    dec_output_thread_.reset();
  }

  if (ctx_ && mpi_) {
    auto ret = mpi_->reset(ctx_);
    if (ret) {
      LOGW(TAG, "Failed to reset decoder ret %d", ret);
    }
  }

  if (packet_) {
    mpp_packet_deinit(&packet_);
    packet_ = nullptr;
  }
  if (frame_) {
    mpp_frame_deinit(&frame_);
    frame_ = nullptr;
  }

  if (ctx_) {
    mpp_destroy(ctx_);
    ctx_ = nullptr;
    mpi_ = nullptr;
  }

  frm_grp_ = nullptr;

  if (buf_mgr_) {
    dec_buf_mgr_deinit(buf_mgr_);
    buf_mgr_ = nullptr;
  }
}

// receive packet & feed to decoder
void MppDecoder::Feed() {
  LOGI(TAG, "feed thread start");

  VideoBuffSlot* packet = nullptr;
  RK_U32 pkt_eos = 0;

  while (!loop_end_.load()) {
    if (encoded_buffer_->IsEmpty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
      continue;
    }

    if (!encoded_buffer_->Dequeue(&packet)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
      continue;
    }

    mpp_packet_set_data(packet_, packet->data);
    mpp_packet_set_size(packet_, packet->size);
    mpp_packet_set_pos(packet_, packet->data);
    mpp_packet_set_length(packet_, packet->size);

    pkt_eos = packet->eos;

    if (pkt_eos) {
      LOGW(TAG, "found eos in packet");
      mpp_packet_set_eos(packet_);
    }

    // send packet until it success
    do {
      auto ret = mpi_->decode_put_packet(ctx_, packet_);
      if (MPP_OK == ret) {
        mpp_assert(0 == mpp_packet_get_length(packet_));
        break;
      }

      // if failed wait a moment and retry
      msleep(1);
    } while (!loop_end_.load());

    // recycle the packet
    buffer_pool_->Release(packet);

    if (pkt_eos) {
      break;
    }
  }

  LOGI(TAG, "feed thread exit");
}

// decode packet & output frame
void MppDecoder::Decode() {
  LOGI(TAG, "decode thread start");

  do {
    RK_U32 frm_eos = 0;
    MppFrame frame = nullptr;

    MPP_RET ret = mpi_->decode_get_frame(ctx_, &frame);
    if (ret) {
      LOGE(TAG, "decode_get_frame failed ret %d", ret);
      continue;
    }

    if (nullptr == frame) {
      msleep(1);
      continue;
    }

    if (mpp_frame_get_info_change(frame)) {
      // found info change and create buffer group for decoding
      RK_U32 width = mpp_frame_get_width(frame);
      RK_U32 height = mpp_frame_get_height(frame);
      RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
      RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
      RK_U32 buf_size = mpp_frame_get_buf_size(frame);

      width_ = width;
      height_ = height;
      format_ = mpp_frame_get_fmt(frame);

      LOGW(TAG, "decode_get_frame get info changed found");
      LOGW(TAG, "decoder require buffer w:h [%d:%d] stride [%d:%d] size %d\n", width, height,
           hor_stride, ver_stride, buf_size);

      if (frm_grp_ == nullptr) {
        if (!CommitBufferGroup(width, height, buf_size)) {
          LOGE(TAG, "failed to setup buffer group for decoder");
          break;
        }
      }

      ret = mpi_->control(ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
      if (ret) {
        LOGE(TAG, "failed to set info change ready");
        break;
      }

      // immediately output the frame
      ret = mpi_->decode_get_frame(ctx_, &frame);
      if (ret) {
        LOGE(TAG, "decode_get_frame failed ret %d", ret);
        continue;
      }
    }

    {
      /*	  char log_buf[256];
          RK_S32 log_size = sizeof(log_buf) - 1;*/
      RK_S32 log_len = 0;
      RK_U32 err_info = mpp_frame_get_errinfo(frame);
      RK_U32 discard = mpp_frame_get_discard(frame);
      int dma_fd = mpp_buffer_get_fd(mpp_frame_get_buffer(frame));

      if (err_info || discard) {
        /*log_len += snprintf(log_buf + log_len, log_size - log_len,
                  " err %x discard %x", err_info, discard);
        log_len += snprintf(log_buf + log_len, log_size - log_len,
                  "decode get frame %d", frame_count_);
        LOGD(TAG, "%s, fd=%d", log_buf, dma_fd);*/
      }

      frame_count_++;

      // save or callback the frame
      if (callback_) {
        callback_(user_data_, mpp_frame_get_hor_stride(frame), mpp_frame_get_ver_stride(frame),
                  mpp_frame_get_width(frame), mpp_frame_get_height(frame), mpp_frame_get_fmt(frame),
                  dma_fd);
      }
    }

    frm_eos = mpp_frame_get_eos(frame);
    mpp_frame_deinit(&frame);

    if (frm_eos) {
      LOGW(TAG, "found eos in frame");
      break;
    }

  } while (!loop_end_.load());

  LOGI(TAG, "decode thread exit");
}

void MppDecoder::PutPacket(const void* data, size_t size, RK_U32 eos) {
  if (use_stream_buffer_) {
    auto packet = buffer_pool_->Acquire(size);
    if (!packet) {
      LOGE(TAG, "Failed to initialize packet from data");
      return;
    }

    if (eos == 1) {
      packet->size = 0;
      packet->eos = 1;
    } else {
      packet->size = (size_t) size;
      std::memcpy(packet->data, data, size);
    }

    if (!encoded_buffer_->Enqueue(packet)) {
      buffer_pool_->Release(packet);
    }
  } else {
    DecodeNow(data, size, eos);
  }
}

void MppDecoder::DecodeNow(const void* data, size_t size, RK_U32 eos) {
  if (eos) {
    LOGW(TAG, "found eos in packet");
    mpp_packet_set_eos(packet_);
  }

  mpp_packet_set_data(packet_, (char*) data);
  mpp_packet_set_size(packet_, size);
  mpp_packet_set_pos(packet_, (char*) data);
  mpp_packet_set_length(packet_, size);

  // send packet until it success
  do {
    auto ret = mpi_->decode_put_packet(ctx_, packet_);
    if (MPP_OK == ret) {
      mpp_assert(0 == mpp_packet_get_length(packet_));
      break;
    }

    // if failed wait a moment and retry
    msleep(1);
  } while (!loop_end_.load());
}

}  // namespace v_dec
