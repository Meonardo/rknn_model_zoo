#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "CameraSource.h"
#include "Common.h"
#include "DetSource.h"
#include "OnvifCore.h"
#include "OnvifSink.h"
#include "RtspSource.h"
#include "SQLiteCpp/SQLiteCpp.h"

#define TAG "App"

#define DEFAULT_DB_DIR "/data/local/tmp/lldb-standalone/"
#define DB_NAME "amdox_facedet.db3"
#define DB_TABLE_NAME "face"

#if USE_QUANTIZED_MODEL
#define DET_MODEL_PATH "/data/local/tmp/lldb-standalone/det_500m_fixed_i8.rknn"
#define EXT_MODEL_PATH "/data/local/tmp/lldb-standalone/w600k_mbf_fixed_i8.rknn"
// #define DET_MODEL_PATH "/data/local/tmp/lldb-standalone/det_2.5g_fixed_i8.rknn"
// #define EXT_MODEL_PATH "/data/local/tmp/lldb-standalone/w600k_r50_fixed_i8.rknn"
#else
#define DET_MODEL_PATH "/data/local/tmp/lldb-standalone/det_500m_fixed_f32.rknn"
#define EXT_MODEL_PATH "/data/local/tmp/lldb-standalone/w600k_mbf_fixed_f32.rknn"
#endif

static int g_mode = 0;  // 0: register, 1: recognize, 2: query registered names(faces), 3: delete
                        // face with(name), 4: clear all registered faces
static char g_name[64] = {0};     // name for registering face
static char g_target[256] = {0};  // image path for registering face or URL(RTSP) for recognition
static char g_db_dir[256] = DEFAULT_DB_DIR;                             // database directory path
static char g_config_dir[256] = "/data/local/tmp/lldb-standalone/bin";  // configuration directory

constexpr float kDetScoreThreshold = 0.5f;
constexpr float kDetNmsThreshold = 0.4f;

struct RtspStream {
  std::string id;
  // Sink
  std::unique_ptr<rtsp::OnvifSink> onvif_sink = nullptr;
  // Sources
  std::unique_ptr<rtsp::RtspSource> video_source = nullptr;
  std::unique_ptr<CameraSource> camera_source = nullptr;
  std::unique_ptr<det::DetSource> det_source = nullptr;

  void Start() {
    if (video_source != nullptr) {
      // Start video source
      video_source->Start();
    }
    // Start detection
    det_source->Start();
    // Link video source to detection source
    if (video_source != nullptr) {
      video_source->AddVideoSink(det_source.get());
    }
    if (camera_source != nullptr) {
      // Link camera source to detection source
      camera_source->AddVideoSink(det_source.get());
    }

    // Link detection source to ONVIF sink
    det_source->AddVideoSink(onvif_sink.get());
  }

  void Stop() {
    if (camera_source != nullptr) {
      // camera_source->Stop();
      camera_source->RemoveVideoSink(det_source.get());
    }
    if (video_source != nullptr) {
      video_source->Stop();
      video_source->RemoveVideoSink(det_source.get());
    }

    det_source->Stop();
    det_source->RemoveVideoSink(onvif_sink.get());

    onvif_sink.reset();
    video_source.reset();
    camera_source.reset();
    det_source.reset();
  }
};

struct App : public rtsp::RtspClientEventHandler {
  std::atomic<bool> is_running{true};

  // Register data
  std::vector<face::FaceLocation> registered_faces;
  // Database
  std::unique_ptr<SQLite::Database> db;
  std::unique_ptr<face::FaceDetector> face_detector;
  std::unique_ptr<face::FaceExtractor> face_extractor;

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

  // stream->camera_source = std::make_unique<CameraSource>("camera_1", "/dev/video14");

  // create detection source and link to video source
  stream->det_source = std::make_unique<det::DetSource>(id, g_app->registered_faces);

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

// Parse command line arguments
// -b <db_dir> : database directory path
// -m <mode> [0-4] : mode 0: register, 1: recognize, 2: query registered
// names(faces), 3: delete face with(name), 4: clear all registered faces
// -n <name> : name for registering face (only for register mode)
// -t <target> : target image path for register or recognize
// -c <config_dir> : configuration directory
// -?h : show help
static void parase_arguments(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-?" || arg == "-h") {
      std::cerr << "Usage: insightface:" << std::endl;
      std::cerr << "  -b <database_dir> : database directory" << std::endl;
      std::cerr << "  -m <mode> : mode 0: register, 1: recognize, 2: query "
                   "registered faces, 3: delete face with(name), 4: clear all "
                   "registered faces "
                << std::endl;
      std::cerr << "  -t <target> : target image path for register or recognize" << std::endl;
      std::cerr << "  -n <name> : name for registering face (only for register mode)" << std::endl;
      std::cerr << "  -c <config_dir> : configuration directory" << std::endl;
      std::cerr << "  -?h : show help" << std::endl;
      exit(0);
    } else if (arg == "-m" && i < argc) {
      g_mode = std::stoi(argv[++i]);
      if (g_mode < 0 || g_mode > 5) {
        std::cerr << "Invalid mode: " << g_mode << std::endl;
        exit(-1);
      }

      std::cerr << "Mode set to: " << g_mode << std::endl;
    } else if (arg == "-b" && i < argc) {
      std::string db_dir = argv[++i];
      strncpy(g_db_dir, db_dir.c_str(), sizeof(g_db_dir) - 1);

      std::cerr << "Database directory set to: " << g_db_dir << std::endl;
    } else if (arg == "-t" && i < argc) {
      std::string target = argv[++i];
      strncpy(g_target, target.c_str(), sizeof(g_target) - 1);

      std::cerr << "Target set to: " << g_target << std::endl;
    } else if (arg == "-n" && i < argc) {
      std::string name = argv[++i];
      strncpy(g_name, name.c_str(), sizeof(g_name) - 1);

      std::cerr << "Registering name: " << name << std::endl;
    } else if (arg == "-c" && i < argc) {
      std::string config_dir = argv[++i];
      strncpy(g_config_dir, config_dir.c_str(), sizeof(g_config_dir) - 1);

      std::cerr << "Configuration directory set to: " << g_config_dir << std::endl;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      exit(-1);
    }
  }
}

/// DATABASE table struct
// id: integer auto increasement
// name: TEXT
// score: FLOAT
// embeddings: BLOB, fixed size (512 floats)
static bool create_db_table(App* app) {
  char statement[128] = {0};
  std::sprintf(statement,
               "CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY "
               "AUTOINCREMENT, name "
               "TEXT, score FLOAT, embeddings BLOB)",
               DB_TABLE_NAME);
  auto ret = app->db->exec(statement);
  return ret >= 0;
}

/// Load registered faces from database into memory
static int load_registered_faces(App* app, std::vector<face::FaceLocation>& faces) {
  SQLite::Statement query(*app->db, "SELECT id, name, score, embeddings FROM " DB_TABLE_NAME);
  int count = 0;
  while (query.executeStep()) {
    face::FaceLocation face;
    face.id = query.getColumn(0).getInt();
    std::strncpy(face.name, query.getColumn(1).getText(), sizeof(face.name) - 1);
    face.score = (float) query.getColumn(2).getDouble();

    // Load embeddings
    const void* blob_data = query.getColumn(3).getBlob();
    int blob_size = query.getColumn(3).getBytes();
    int num_floats = blob_size / sizeof(float);
    face.embedding.resize(num_floats);
    std::memcpy(face.embedding.data(), blob_data, blob_size);

    faces.push_back(std::move(face));
    count++;
  }
  return count;
}

/// Check if a face with the given name already exists in the database
static bool check_face_exists(App* app, const char* name) {
  SQLite::Statement query(*app->db, "SELECT COUNT(*) FROM " DB_TABLE_NAME " WHERE name = ?");
  query.bind(1, name);
  if (query.executeStep()) {
    int count = query.getColumn(0).getInt();
    return count > 0;
  }
  return false;
}

/// Insert face data into database
/// return the id of the inserted row, or -1 on failure
static int insert_face_to_db(App* app, const face::FaceLocation& face) noexcept {
  SQLite::Statement query(
      *app->db, "INSERT INTO " DB_TABLE_NAME " (name, score, embeddings) VALUES (?, ?, ?)");
  query.bind(1, face.name);
  query.bind(2, face.score);
  query.bind(3, face.embedding.data(), static_cast<int>(face.embedding.size() * sizeof(float)));
  auto ret = query.exec();
  if (ret > 0) {
    return static_cast<int>(app->db->getLastInsertRowid());
  }

  return -1;
}

static rga_buffer_t* create_rga_buffer_from_mat(const cv::Mat& mat) {
  rga_buffer_t* rga_buf = new rga_buffer_t;
  memset(rga_buf, 0, sizeof(rga_buffer_t));

  int frame_width = mat.cols;
  int frame_height = mat.rows;

  // Allocate buffer for source image
  auto buf_size = frame_width * frame_height * 3;
  int fd = -1;
  char* vir_addr = (char*) calloc(buf_size, sizeof(char));
  if (vir_addr == nullptr) {
    assert(false && "FaceExtractor alloc system buffer failed");
  }
  // Import to RGA buffer
  auto handle = importbuffer_virtualaddr(vir_addr, buf_size);
  if (handle == 0) {
    assert(false && "FaceExtractor importbuffer_virtualaddr failed");
  }
  *rga_buf = wrapbuffer_handle(handle, frame_width, frame_height, RK_FORMAT_RGB_888);
  rga_buf->vir_addr = vir_addr;

  // Copy src to `rga_buf`
  std::memcpy(rga_buf->vir_addr, mat.data, buf_size);

  return rga_buf;
}

static void destroy_rga_buffer(rga_buffer_t* rga_buf) {
  if (rga_buf) {
    releasebuffer_handle(rga_buf->handle);
    free(rga_buf->vir_addr);
    delete rga_buf;
  }
}

static void draw_boxes(cv::Mat& mat, const std::vector<face::FaceLocation>& faces) {
  for (const auto& face : faces) {
    cv::rectangle(mat, face.box, cv::Scalar(0, 255, 0), 2);
    std::string label = face.name;
    if (label.empty()) {
      label = "ID:" + std::to_string(face.id);
    }
    int x = face.box.x;
    int y = face.box.y;
    cv::putText(mat, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 2);
  }

  cv::imwrite("/sdcard/Download/det_result.jpg", mat);
}

static int register_face(App* app, const std::string& name, const std::string& image_path) {
  // Load image
  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return -1;
  }

  // Init face detector & extractor
  app->face_detector =
      std::make_unique<face::FaceDetector>(DET_MODEL_PATH, kDetScoreThreshold, kDetNmsThreshold);
  app->face_extractor = std::make_unique<face::FaceExtractor>(EXT_MODEL_PATH, img.cols, img.rows);

  // Check if the face with the same name already exists
  for (const auto& face : app->registered_faces) {
    if (std::strncmp(face.name, name.c_str(), 64) == 0) {
      std::cerr << "A face with the name \"" << name << "\" is already registered." << std::endl;
      return 0;
    }
  }
  // Alternatively, check in the database
  if (check_face_exists(g_app.get(), name.c_str())) {
    std::cerr << "A face with the name \"" << name << "\" is already registered in the database."
              << std::endl;
    return 0;
  }

  // Convert BGR to RGB
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

  // Create RGA buffer from image
  auto rga_buf = create_rga_buffer_from_mat(rgb_img);
  // Detect faces
  auto faces = app->face_detector->Detect(rga_buf);
  if (faces.empty()) {
    std::cerr << "No face detected in the image: " << image_path << std::endl;
    destroy_rga_buffer(rga_buf);
    return -1;
  }

  // Use the first detected face as register face
  auto& face = faces[0];
  // Extract embedding
  auto ret = g_app->face_extractor->Extract(rga_buf, faces);
  if (ret < 0) {
    std::cerr << "Failed to extract face embedding." << std::endl;
    destroy_rga_buffer(rga_buf);
    return -1;
  }

  // Insert into database
  face.score = 0.5555f;  // dummy score
  std::strncpy(face.name, name.c_str(), sizeof(face.name) - 1);
  ret = insert_face_to_db(g_app.get(), face);
  if (ret < 0) {
    std::cerr << "Failed to insert face into database." << std::endl;
  } else {
    std::cerr << "Successfully registered face \"" << name << "\" with ID " << ret << std::endl;
    // Draw box and label on image
    draw_boxes(img, {face});
  }

  destroy_rga_buffer(rga_buf);

  return ret;
}

static int recognize_face(App* app, const std::string& target) {
  // Register signal handlers for graceful shutdown
  struct sigaction sa = {};
  sa.sa_handler = handle_sig;
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  std::string rtsp_url = g_target;
  if (rtsp_url.find("rtsp://") != 0) {
    std::cout << "Invalid RTSP URL: " << rtsp_url << std::endl;
    return -1;
  }
  std::string config_dir = g_config_dir;
  if (!std::filesystem::exists(config_dir) || !std::filesystem::is_directory(config_dir)) {
    std::cout << "Invalid config directory: " << config_dir << std::endl;
    return -1;
  }

  // Initialize ONVIF core
  app->onvif_core = std::make_unique<rtsp::OnvifCore>(config_dir, app);

  // Get RTSP stream configurations
  auto rtsp_cfgs = app->onvif_core->GetRtspStreams();
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

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Entry point
int main(int argc, char** argv) {
  parase_arguments(argc, argv);

  // Initialize application
  g_app = std::make_unique<App>();
  // Init database
  auto db_dir_str = std::filesystem::path(g_db_dir) / DB_NAME;
  g_app->db =
      std::make_unique<SQLite::Database>(db_dir_str, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
  if (!create_db_table(g_app.get())) {
    std::cerr << "Failed to create database table" << std::endl;
    return -1;
  }
  // Load registered faces from database
  if (load_registered_faces(g_app.get(), g_app->registered_faces) < 0) {
    std::cerr << "Failed to load registered faces." << std::endl;
    return -2;
  }
  for (auto& face : g_app->registered_faces) {
    LOGI(TAG, "Loaded registered face: id=%d, name=%s, score=%.4f", face.id, face.name,
         face.recog_score);
  }

  if (g_mode == 0) {
    // Register face
    return register_face(g_app.get(), g_name, g_target);
  } else if (g_mode == 1) {
    // Recognize face
    return recognize_face(g_app.get(), g_target);
  } else {
    std::cerr << "Mode " << g_mode << " not implemented yet." << std::endl;
    return -1;
  }

  g_app.reset();

  return 0;
}