//
// Created by Meonardo on 11/18/2024.
//

#ifndef ONVIFCORE_H_
#define ONVIFCORE_H_

#include <string>
#include <vector>

#include "VideoCommon.h"

namespace rtsp {

class OnvifCore {
 public:
  explicit OnvifCore(std::string_view config_dir,
                     rtsp::RtspClientEventHandler* rtsp_client_event_handler);
  ~OnvifCore();

  const std::vector<RtspStreamCfg> GetRtspStreams() const {
    return rtsp_stream_cfgs_;
  }

 private:
  std::string config_dir_;
  rtsp::RtspClientEventHandler* rtsp_client_event_handler_;
  std::vector<RtspStreamCfg> rtsp_stream_cfgs_;

  void Init();
  void DeInit();
  void OnRtspClientStateChanged(uint32_t msg_sub_type, std::string_view client_id,
                                std::string_view client_ip, uint16_t client_port);
};

}  // namespace rtsp

#endif  // ONVIFCORE_H_
