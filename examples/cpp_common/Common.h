//
// Created by Meonardo on 8/13/2025.
//

#ifndef COMMON_H_
#define COMMON_H_

#include <assert.h>
#include <sys/time.h>

#if ANDROID_APK
#include <android/log.h>
#include <jni.h>

/// defines
#define JNI_EXPORT extern "C" JNIEXPORT JNICALL

#define blog(level, tag, msg, ...) __android_log_print(level, tag, msg, ##__VA_ARGS__)

#define LOGE(tag, ...) __android_log_print(ANDROID_LOG_ERROR, tag, __VA_ARGS__)
#define LOGI(tag, ...) __android_log_print(ANDROID_LOG_INFO, tag, __VA_ARGS__)
#define LOGD(tag, ...) __android_log_print(ANDROID_LOG_DEBUG, tag, __VA_ARGS__)
#define LOGW(tag, ...) __android_log_print(ANDROID_LOG_WARN, tag, __VA_ARGS__)

#define FUN_BEGIN_TIME(FUN)                          \
  {                                                  \
    LOGD("TIME", "%s:%s func start", __FILE__, FUN); \
    long long t0 = GetSysCurrentTime();

#define FUN_END_TIME(FUN)                                                      \
  long long t1 = GetSysCurrentTime();                                          \
  LOGD("TIME", "%s:%s func cost time %ldms", __FILE__, FUN, (long) (t1 - t0)); \
  }

static long long GetSysCurrentTime() {
  struct timeval time = {0};
  gettimeofday(&time, NULL);
  long long curTime = ((long long) (time.tv_sec)) * 1000 + time.tv_usec / 1000;
  return curTime;
}
#else
#include <time.h>

//-----------------------------------------------------------------------------
// Returns a pointer to a static string buffer with the current local time
// formatted as "yyyy-MM-dd HH:mm:ss.SSS".
//
// WARNING: This uses a single static buffer. Not thread-safe for concurrent
// calls!
//-----------------------------------------------------------------------------
static inline const char* getFormattedTimestamp() {
  static char buffer[64];  // Enough space for "yyyy-MM-dd HH:mm:ss.SSS"
  struct timeval tv;
  gettimeofday(&tv, NULL);

  time_t nowSec = tv.tv_sec;
  struct tm lt;
  localtime_r(&nowSec, &lt);

  // Format date/time: "yyyy-MM-dd HH:mm:ss"
  // (strftime leaves out the milliseconds)
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &lt);

  // Append milliseconds
  int ms = (int) (tv.tv_usec / 1000);  // microseconds -> milliseconds
  char msBuf[8];
  snprintf(msBuf, sizeof(msBuf), ".%03d", ms);

  // Safe string concatenation
  strncat(buffer, msBuf, sizeof(buffer) - strlen(buffer) - 1);

  return buffer;
}

// ANSI escape codes for colors
#define COLOR_RED "\x1b[31;1m"
#define COLOR_GREEN "\x1b[32;1m"
#define COLOR_YELLOW "\x1b[33;1m"
#define COLOR_CYAN "\x1b[36;1m"
#define COLOR_RESET "\x1b[0m"

//----------------------------------------------------------------------------
// Logging macros (Linux / Unix)
//----------------------------------------------------------------------------

#define LOGE(tag, fmt, ...)                                                                     \
  do {                                                                                          \
    fprintf(stderr, COLOR_RED "[%s][ERROR][%s] " fmt "\n" COLOR_RESET, getFormattedTimestamp(), \
            tag, ##__VA_ARGS__);                                                                \
  } while (0)

#define LOGW(tag, fmt, ...)                                                                        \
  do {                                                                                             \
    fprintf(stderr, COLOR_YELLOW "[%s][WARN ][%s] " fmt "\n" COLOR_RESET, getFormattedTimestamp(), \
            tag, ##__VA_ARGS__);                                                                   \
  } while (0)

#define LOGI(tag, fmt, ...)                                                                       \
  do {                                                                                            \
    fprintf(stdout, COLOR_GREEN "[%s][INFO ][%s] " fmt "\n" COLOR_RESET, getFormattedTimestamp(), \
            tag, ##__VA_ARGS__);                                                                  \
  } while (0)

#define LOGD(tag, fmt, ...)                                                                      \
  do {                                                                                           \
    fprintf(stdout, COLOR_CYAN "[%s][DEBUG][%s] " fmt "\n" COLOR_RESET, getFormattedTimestamp(), \
            tag, ##__VA_ARGS__);                                                                 \
  } while (0)
#endif

#endif  // COMMON_H_
