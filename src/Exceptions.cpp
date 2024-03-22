#include "Exceptions.h"
#include <sstream>

namespace pathtracex {

	PathException::PathException(int line, const char* file) noexcept  :
		line(line),
		file(file)
	{ }

	const char* PathException::what() const noexcept {
		std::ostringstream oss;
		oss << getType() << '\n'
			<< getOriginString() << std::endl;
		whatBuffer = oss.str();
		return whatBuffer.c_str();
	}

	const char* PathException::getType() const noexcept {
		return "PathException";
	}

	int PathException::getLine() const noexcept {
		return line;
	}

	const std::string& PathException::getFile() const noexcept {
		return file;
	}

	std::string PathException::getOriginString() const noexcept {
		std::ostringstream oss;
		oss << "[File] " << file << std::endl
			<< "[Line] " << line;
		return oss.str();
	}

	HRException::HRException(int line, const char* file, HRESULT hr) :
		PathException(line, file),
		hr(hr)
	{ }

	std::string HRException::getErrorMessageFromHResult() const noexcept {
		char buffer[512];
		auto success = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, hr,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			buffer, _countof(buffer), nullptr);

		if (success) {
			return std::string(buffer);
		}

		return "";
	}

	/*
	const wchar_t* ErrorMessage() const throw()
    {
        if (m_message == nullptr)
        {
            wchar_t buffer[4096];
            if (::FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM,
                                    nullptr, 
                                    m_hr,
                                    MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL),
                                    buffer, 
                                    _countof(buffer), 
                                    nullptr))
            {
                m_message = AllocateString(buffer);
            }
        }
        return m_message;
    }
	*/
	const char* HRException::what() const noexcept {
		std::ostringstream oss;
		oss << getType() << '\n'
			<< getOriginString() << '\n'
			<< "[Error] " << getErrorMessageFromHResult() << std::endl;
		whatBuffer = oss.str();
		return whatBuffer.c_str();
	}

	const char* HRException::getType() const noexcept {
		return "HRException";
	}

}