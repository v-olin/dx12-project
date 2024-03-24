#pragma once

#include <spdlog/spdlog.h>
#include <memory>

namespace pathtracex {
	class Logger {
	public:
		static void init();

		inline static std::shared_ptr<spdlog::logger>& getLogger() { return logger; }

	private:
		static std::shared_ptr<spdlog::logger> logger;
	};
}

#if defined(_DEBUG) && defined(ENABLE_TRACE_LOGGING)
#define LOG_TRACE(...) Logger::getLogger()->trace("Trace: " __VA_ARGS__ ) // Only log trace in debug mode
#else
#define LOG_TRACE(...)
#endif

#define LOG_INFO(...)  Logger::getLogger()->info("Info: " __VA_ARGS__ )
#define LOG_WARN(...)  Logger::getLogger()->warn("Warn: " __VA_ARGS__ )
#define LOG_ERROR(...) Logger::getLogger()->error("Error: " __VA_ARGS__ )
#define LOG_FATAL(...) Logger::getLogger()->critical("Fatal: " __VA_ARGS__ )
#define LOG_CRITICAL(...) Logger::getLogger()->critical("Critical: " __VA_ARGS__ )