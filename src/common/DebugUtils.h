/**
 * @file DebugUtils.h
 * @brief Structured logging utilities for consistent debugging across the codebase.
 *
 * Provides macros and a Logger class for debug output that can be:
 * - Disabled completely in release builds (when NDEBUG is defined)
 * - Controlled at runtime via environment variables in debug builds
 *
 * **Usage:**
 *
 * @code
 * #include "DebugUtils.h"
 *
 * void some_function() {
 *     // Simple debug logging (disabled in release builds)
 *     DEBUG_LOG("Entering function");
 *     DEBUG_LOG("Value: " << some_value);
 *     DEBUG_LOG_IF(condition, "Conditional message");
 *
 *     // Structured logging with levels (always available)
 *     LOG_DEBUG("Detailed info for debugging");
 *     LOG_INFO("Informational message");
 *     LOG_WARN("Warning message");
 *     LOG_ERROR("Error message");
 * }
 * @endcode
 *
 * **Log Level Control:**
 *
 * Set the FTS_LOG_LEVEL environment variable to control output:
 * - FTS_LOG_LEVEL=0 : No logging
 * - FTS_LOG_LEVEL=1 : Errors only
 * - FTS_LOG_LEVEL=2 : Errors + Warnings
 * - FTS_LOG_LEVEL=3 : Errors + Warnings + Info (default)
 * - FTS_LOG_LEVEL=4 : All (including Debug)
 *
 * **Build Modes:**
 *
 * - Debug build: DEBUG_* macros active, LOG_DEBUG active
 * - Release build (NDEBUG defined): DEBUG_* macros expand to nothing,
 *   LOG_DEBUG is disabled but LOG_INFO/WARN/ERROR remain available
 *
 * @note Use sparingly in hot paths as logging affects performance.
 */

#ifndef DEBUG_UTILS_H_
#define DEBUG_UTILS_H_

#include <iostream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <mutex>

namespace logging {

/**
 * @brief Log levels for structured logging.
 */
enum class LogLevel {
    NONE = 0,    ///< No logging
    ERROR = 1,   ///< Errors only
    WARN = 2,    ///< Warnings and above
    INFO = 3,    ///< Info and above (default)
    DEBUG = 4    ///< All messages including debug
};

/**
 * @brief Thread-safe logger with configurable log level.
 *
 * Log level is controlled by FTS_LOG_LEVEL environment variable.
 */
class Logger {
public:
    /**
     * @brief Get the singleton logger instance.
     */
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    /**
     * @brief Log a message at the specified level.
     *
     * @param level Log level for this message
     * @param msg   Message to log
     * @param file  Source file name (optional)
     * @param line  Source line number (optional)
     */
    void log(LogLevel level, const std::string& msg,
             const char* file = nullptr, int line = 0) {
        if (static_cast<int>(level) > static_cast<int>(current_level_)) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        std::cerr << "[" << level_string(level) << "]";
        if (file != nullptr) {
            std::cerr << " " << file << ":" << line;
        }
        std::cerr << " " << msg << std::endl;
    }

    /**
     * @brief Check if a given log level is enabled.
     */
    bool is_enabled(LogLevel level) const {
        return static_cast<int>(level) <= static_cast<int>(current_level_);
    }

    /**
     * @brief Get the current log level.
     */
    LogLevel get_level() const { return current_level_; }

    /**
     * @brief Set the log level programmatically.
     */
    void set_level(LogLevel level) { current_level_ = level; }

private:
    Logger() {
        // Initialize from environment variable
        const char* env = std::getenv("FTS_LOG_LEVEL");
        if (env != nullptr) {
            int level = std::atoi(env);
            if (level >= 0 && level <= 4) {
                current_level_ = static_cast<LogLevel>(level);
            }
        }
    }

    static const char* level_string(LogLevel level) {
        switch (level) {
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::DEBUG: return "DEBUG";
            default:              return "?????";
        }
    }

    LogLevel current_level_ = LogLevel::INFO;
    std::mutex mutex_;
};

} // namespace logging

// =============================================================================
// Structured logging macros (always available, respects FTS_LOG_LEVEL)
// =============================================================================

/**
 * @brief Log an error message.
 */
#define LOG_ERROR(msg) \
    do { \
        if (logging::Logger::instance().is_enabled(logging::LogLevel::ERROR)) { \
            std::ostringstream oss; \
            oss << msg; \
            logging::Logger::instance().log(logging::LogLevel::ERROR, oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

/**
 * @brief Log a warning message.
 */
#define LOG_WARN(msg) \
    do { \
        if (logging::Logger::instance().is_enabled(logging::LogLevel::WARN)) { \
            std::ostringstream oss; \
            oss << msg; \
            logging::Logger::instance().log(logging::LogLevel::WARN, oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

/**
 * @brief Log an info message.
 */
#define LOG_INFO(msg) \
    do { \
        if (logging::Logger::instance().is_enabled(logging::LogLevel::INFO)) { \
            std::ostringstream oss; \
            oss << msg; \
            logging::Logger::instance().log(logging::LogLevel::INFO, oss.str()); \
        } \
    } while(0)

#ifndef NDEBUG
/**
 * @brief Log a debug message (only in debug builds).
 */
#define LOG_DEBUG(msg) \
    do { \
        if (logging::Logger::instance().is_enabled(logging::LogLevel::DEBUG)) { \
            std::ostringstream oss; \
            oss << msg; \
            logging::Logger::instance().log(logging::LogLevel::DEBUG, oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)
#else
#define LOG_DEBUG(msg) do {} while(0)
#endif

// =============================================================================
// Legacy debug macros (disabled in release builds)
// =============================================================================

#ifndef NDEBUG

/**
 * @brief Debug log macro - outputs message with file and line info.
 *
 * Only active in debug builds. In release builds, expands to nothing.
 *
 * @param msg Message to output (can use << for streaming)
 *
 * @example
 * @code
 * DEBUG_LOG("Processing " << n << " elements");
 * @endcode
 */
#define DEBUG_LOG(msg) \
    do { \
        std::cerr << "[DEBUG] " << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; \
    } while(0)

/**
 * @brief Conditional debug log macro.
 *
 * Only outputs if condition is true. Only active in debug builds.
 *
 * @param cond Condition to check
 * @param msg  Message to output
 *
 * @example
 * @code
 * DEBUG_LOG_IF(value > threshold, "Value " << value << " exceeds threshold");
 * @endcode
 */
#define DEBUG_LOG_IF(cond, msg) \
    do { \
        if (cond) { \
            std::cerr << "[DEBUG] " << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; \
        } \
    } while(0)

/**
 * @brief Simple debug log without file/line info.
 *
 * Useful for cleaner output in verbose debugging sections.
 *
 * @param msg Message to output
 */
#define DEBUG_PRINT(msg) \
    do { \
        std::cerr << msg << std::endl; \
    } while(0)

/**
 * @brief Debug block - executes code only in debug builds.
 *
 * @example
 * @code
 * DEBUG_BLOCK({
 *     for (int i = 0; i < 10; i++) {
 *         std::cerr << "value[" << i << "] = " << values[i] << std::endl;
 *     }
 * });
 * @endcode
 */
#define DEBUG_BLOCK(code) do { code } while(0)

#else // NDEBUG defined - release build

// In release builds, all debug macros expand to nothing
#define DEBUG_LOG(msg) do {} while(0)
#define DEBUG_LOG_IF(cond, msg) do {} while(0)
#define DEBUG_PRINT(msg) do {} while(0)
#define DEBUG_BLOCK(code) do {} while(0)

#endif // NDEBUG

namespace debug {

/**
 * @brief Check if debug mode is enabled.
 *
 * @return true if compiled in debug mode (NDEBUG not defined)
 */
inline bool is_debug_build()
{
#ifndef NDEBUG
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get build type as string.
 *
 * @return "Debug" or "Release"
 */
inline std::string build_type()
{
#ifndef NDEBUG
    return "Debug";
#else
    return "Release";
#endif
}

} // namespace debug

#endif // DEBUG_UTILS_H_
