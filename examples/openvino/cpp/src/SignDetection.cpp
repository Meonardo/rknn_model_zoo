#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "Common.h"
#include "OnvifCore.h"
#include "OnvifSink.h"
#include "RtspSource.h"
#include "DetSource.h"

#define TAG "App"

struct RtspStream {
  std::string id;
  // Sink
  std::unique_ptr<rtsp::OnvifSink> onvif_sink = nullptr;
  // Sources
  std::unique_ptr<rtsp::RtspSource> video_source = nullptr;
  std::unique_ptr<det::DetSource> det_source = nullptr;

  void Start() {
    // Start video source
    video_source->Start();
    // Start detection
    det_source->Start();
    // Link video source to detection source
    video_source->AddVideoSink(det_source.get());
    // Link detection source to ONVIF sink
    det_source->AddVideoSink(onvif_sink.get());
  }

  void Stop() {
    video_source->Stop();
    video_source->RemoveVideoSink(det_source.get());
    det_source->Stop();
    det_source->RemoveVideoSink(onvif_sink.get());

    onvif_sink.reset();
    video_source.reset();
    det_source.reset();
  }
};

struct App : public rtsp::RtspClientEventHandler {
  std::atomic<bool> is_running{true};

  // ONVIF
  std::unique_ptr<rtsp::OnvifCore> onvif_core;

  std::vector<std::unique_ptr<RtspStream>> rtsp_streams;

  void OnRtspClientStateChanged(std::string_view id, rtsp::RtspMessageType msg_sub_type,
                                std::string_view client_ip, uint16_t client_port) override {
    if (msg_sub_type == rtsp::RtspMessageType::RTSP_MT_PLAY) {
      auto it = std::find_if(
          rtsp_streams.begin(), rtsp_streams.end(),
          [&id](const std::unique_ptr<RtspStream>& instance) { return instance->id == id; });
      if (it != rtsp_streams.end()) {
        LOGI(TAG, "RTSP client %s connected from %s:%d", id.data(), client_ip.data(), client_port);
        // it->get()->encoded_video_source->RequestKeyFrame();
      } else {
        LOGW(TAG, "RTSP client %s connected but instance not found", id.data());
      }
    } else if (msg_sub_type == rtsp::RtspMessageType::RTSP_MT_TEARDOWN) {
      LOGI(TAG, "RTSP client %s disconnected from %s:%d", id.data(), client_ip.data(), client_port);
    } else {
      LOGW(TAG, "Unhandled RTSP message type: %d for client %s", static_cast<int>(msg_sub_type),
           id.data());
    }
  }
};
// Global application instance
static std::unique_ptr<App> g_app;

static void create_rtsp_stream(std::string_view id, std::string_view src_url,
                               const rtsp::RtspStreamCfg& cfg) {
  auto stream = std::make_unique<RtspStream>();
  stream->id = id;

  // create ONVIF sink
  stream->onvif_sink = std::make_unique<rtsp::OnvifSink>(id, cfg);

  // create RTSP source
  stream->video_source = std::make_unique<rtsp::RtspSource>(id, src_url);

  // create detection source and link to video source
  stream->det_source = std::make_unique<det::DetSource>(id);

  // start streaming
  stream->Start();

  g_app->rtsp_streams.push_back(std::move(stream));
}

static void destroy_rtsp_streams() {
  for (auto& stream : g_app->rtsp_streams) {
    stream->Stop();
  }
  g_app->rtsp_streams.clear();
}

// handle abnormal termination signals
static void handle_sig(int32_t signo) {
  if (signo == SIGINT || signo == SIGTERM) {
    // handle graceful shutdown
    LOGI(TAG, "Received signal %d, shutting down...", signo);
    g_app->is_running.store(false);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Entry point
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <rtsp_url> <config_dir>" << std::endl;
    return -1;
  }

  std::string rtsp_url = argv[1];
  if (rtsp_url.find("rtsp://") != 0) {
    std::cout << "Invalid RTSP URL: " << rtsp_url << std::endl;
    return -1;
  }
  std::string config_dir = argv[2];
  if (!std::filesystem::exists(config_dir) || !std::filesystem::is_directory(config_dir)) {
    std::cout << "Invalid config directory: " << config_dir << std::endl;
    return -1;
  }

  // Initialize application
  g_app = std::make_unique<App>();
  // Initialize ONVIF core
  g_app->onvif_core = std::make_unique<rtsp::OnvifCore>(config_dir, g_app.get());

  // Register signal handlers for graceful shutdown
  struct sigaction sa = {};
  sa.sa_handler = handle_sig;
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  // Get RTSP stream configurations
  auto rtsp_cfgs = g_app->onvif_core->GetRtspStreams();
  if (rtsp_cfgs.empty()) {
    LOGE(TAG, "No RTSP streams configured, exiting.");
    return -1;
  }

  // Create RTSP streams
  const auto& rtsp_cfg = rtsp_cfgs[0];
  std::string stream_id = std::to_string(1);
  create_rtsp_stream(stream_id, rtsp_url, rtsp_cfg);

  // Main loop
  while (g_app->is_running.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Cleanup
  destroy_rtsp_streams();
  g_app->onvif_core.reset();
  g_app.reset();

  return 0;
}