/**
 * @file DebugUtils.h
 * @brief Debug output utilities for consistent debugging across the codebase.
 *
 * Provides macros and functions for debug output that are automatically
 * disabled in release builds (when NDEBUG is defined).
 *
 * **Usage:**
 *
 * @code
 * #include "DebugUtils.h"
 *
 * void some_function() {
 *     DEBUG_LOG("Entering function");
 *     DEBUG_LOG("Value: " << some_value);
 *     DEBUG_LOG_IF(condition, "Conditional message");
 * }
 * @endcode
 *
 * **Build Modes:**
 *
 * - Debug build: All debug output is printed to stderr
 * - Release build (NDEBUG defined): All debug macros expand to nothing
 *
 * @note Use sparingly in hot paths as debug output affects performance.
 */

#ifndef DEBUG_UTILS_H_
#define DEBUG_UTILS_H_

#include <iostream>
#include <string>

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
