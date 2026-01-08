//
// Created by Meonardo on 11/18/2024.
//

#include "OnvifCore.h"

// onvif headers
#include <vector>

#include "Common.h"
#include "http_parse.h"
#include "live_video.h"
#include "media_format.h"
#include "onvif.h"
#include "onvif_srv.h"
#include "rtsp_cfg.h"
#include "rtsp_srv.h"
#include "sys_inc.h"

#define TAG "OnvifCore"

#define ONVIF_CONFIG_FILE "onvif.cfg"
#define RTSP_CONFIG_FILE "rtsp.cfg"
#define MAX_FILE_PATH_LENGTH 256

constexpr int ONVIF_PORT = 8000;
constexpr int MAX_HTTP_USER = 16;
constexpr int RTSP_PORT = 8554;
constexpr int RTSP_HTTP_PORT = 8080;

extern "C" ONVIF_CLS g_onvif_cls;

static void set_onvif_rtsp_port() {
  g_onvif_cls.rtsp_port = g_rtsp_cfg.serverport;
}

namespace rtsp {

OnvifCore::OnvifCore(std::string_view config_dir, rtsp::RtspClientEventHandler* handler)
    : config_dir_(config_dir), rtsp_client_event_handler_(handler) {
  LOGI(TAG, "Call create OnvifCore, loading configuration from: %s", config_dir_.c_str());

  Init();
}

OnvifCore::~OnvifCore() {
  LOGI(TAG, "Call destroy OnvifCore");

  DeInit();
}

void OnvifCore::Init() {
  sys_buf_init(MAX_NUM_RUA * 2 + 32);
  http_msg_buf_init(MAX_NUM_RUA * 2 + 16);

  // init & start ONVIF
  char onvif_cfg_path[MAX_FILE_PATH_LENGTH] = {0};
  snprintf(onvif_cfg_path, sizeof(onvif_cfg_path), "%s/%s", config_dir_.c_str(), ONVIF_CONFIG_FILE);
  onvif_start(onvif_cfg_path);

  // init & start RTSP server
  char rtsp_cfg_path[MAX_FILE_PATH_LENGTH] = {0};
  snprintf(rtsp_cfg_path, sizeof(rtsp_cfg_path), "%s/%s", config_dir_.c_str(), RTSP_CONFIG_FILE);
  if (!rtsp_start(
          rtsp_cfg_path,
          +[](uint32_t msg_sub_type, const char* path, uint32_t client_ip, uint16_t client_port,
              void* user_data) {
            if (path == nullptr || user_data == nullptr) {
              LOGE(TAG, "Invalid parameters in RTSP server callback");
              return;
            }
            std::string stream_name(path);
            auto onvif_core = static_cast<OnvifCore*>(user_data);
            // convert uint32_t to ip string
            char client_ip_str[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &client_ip, client_ip_str, sizeof(client_ip_str));
            onvif_core->OnRtspClientStateChanged(msg_sub_type, stream_name, client_ip_str,
                                                 client_port);
          },
          this)) {
    LOGE(TAG, "Failed to start RTSP server with config: %s", rtsp_cfg_path);
    return;
  }

  set_onvif_rtsp_port();

  // Try to get RTSP server configuration, like outputs
  int output_count = rtsp_cfg_get_output_count();
  if (output_count > 0) {
    MEDIA_OUTPUT* outputs = new MEDIA_OUTPUT[output_count];
    auto success = rtsp_cfg_get_all_outputs(outputs);
    if (success) {
      for (int i = 0; i < output_count; i++) {
        RtspStreamCfg rtsp_cfg{};
        rtsp_cfg.url = outputs[i].url;

        rtsp_cfg.video_info.width = outputs[i].video.width;
        rtsp_cfg.video_info.height = outputs[i].video.height;
        rtsp_cfg.video_info.bitrate = outputs[i].video.bitrate * 1000;  // convert from kbps to bps
        rtsp_cfg.video_info.fps = static_cast<uint16_t>(outputs[i].video.framerate);
        if (outputs[i].video.codec == VIDEO_CODEC_H264) {
          rtsp_cfg.video_info.codec = 0;  // h264
          rtsp_cfg.video_info.codec_name = "h264";
          rtsp_cfg.video_info.stream_format = "annexb";
          rtsp_cfg.video_info.profile_level_id = "4D0029";  // high profile
          rtsp_cfg.video_info.alignment = "au";             // or "nal"
          rtsp_cfg.video_info.pixel_format = 0;             // nv12
          rtsp_cfg.video_info.rate_control = 1;             // CBR
        } else if (outputs[i].video.codec == VIDEO_CODEC_H265) {
          rtsp_cfg.video_info.codec = 1;  // h265
          rtsp_cfg.video_info.codec_name = "h265";
          rtsp_cfg.video_info.stream_format = "annexb";
          rtsp_cfg.video_info.profile_level_id = "018028";  // main10 profile
          rtsp_cfg.video_info.alignment = "au";             // or "nal"
          rtsp_cfg.video_info.pixel_format = 0;             // nv12
          rtsp_cfg.video_info.rate_control = 1;             // CBR
        } else {
          LOGE(TAG, "Unsupported video codec: %d", outputs[i].video.codec);
          continue;
        }

        rtsp_stream_cfgs_.emplace_back(std::move(rtsp_cfg));

        LOGI(TAG, "Loaded RTSP stream: URL:[%s], video: %dx%d@%ffps", outputs[i].url,
             outputs[i].video.width, outputs[i].video.height, outputs[i].video.framerate);
      }
    }
    delete[] outputs;
  }
}

void OnvifCore::DeInit() {
  // stop rtsp server
  rtsp_stop();
  // stop onvif
  onvif_stop();

  // release resourcess
  sys_buf_deinit();
  http_msg_buf_deinit();
}

void OnvifCore::OnRtspClientStateChanged(uint32_t msg_sub_type, std::string_view client_id,
                                         std::string_view client_ip, uint16_t client_port) {
  rtsp_client_event_handler_->OnRtspClientStateChanged(
      client_id, (rtsp::RtspMessageType) msg_sub_type, client_ip, client_port);
}

}  // namespace rtsp