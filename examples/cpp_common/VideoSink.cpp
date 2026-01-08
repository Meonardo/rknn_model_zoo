//
// Created by Meonardo on 8/20/2024.
//

#include "VideoSink.h"

EncodedVideoInfo::EncodedVideoInfo()
    : width(kBaseVideoWidth),
      height(kBaseVideoHeight),
      bitrate(2000000),
      fps(kDefaultFps),
      codec(0),
	  rate_control(0),
      codec_name("H264"),
      stream_format("byte-stream"),
      profile_level_id("4D0029"),
      alignment("au"), pixel_format(0) {}