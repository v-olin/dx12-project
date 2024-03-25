#include <iostream>
#include "Logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace pathtracex {

	void Logger::init() {
		logger = spdlog::stdout_color_mt("LOGGER");
		spdlog::set_default_logger(logger);
		spdlog::set_pattern("%^[%T] %n: %v%$");
		logger->set_level(spdlog::level::trace);
		
		LOG_INFO("Logger initialized");
	}

	std::shared_ptr<spdlog::logger> Logger::logger;

}
