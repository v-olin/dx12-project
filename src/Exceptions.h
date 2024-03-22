#pragma once

#include <exception>
#include <string>

#include "PathWin.h"

#define THROW_IF_FAILED(hrcall) if(FAILED(hr = (hrcall))) { throw HRException(__LINE__, __FILE__, hr); }

namespace pathtracex {

	class PathException : public std::exception {
	public:
		PathException(int line, const char* file) noexcept;
		
		const char* what() const noexcept override;
		virtual const char* getType() const noexcept;
		int getLine() const noexcept;
		const std::string& getFile() const noexcept;
		std::string getOriginString() const noexcept;

	protected:
		int line;
		std::string file;

	private:
		mutable std::string whatBuffer;
	};

	class HRException : public PathException {
	public:
		HRException(int line, const char* file, HRESULT hr);

		const char* what() const noexcept override;
		const char* getType() const noexcept override;
	private:
		std::string getErrorMessageFromHResult() const noexcept;

		HRESULT hr;
		mutable std::string whatBuffer;
	};

}