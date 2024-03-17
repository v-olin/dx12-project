#pragma once

#include "PathWin.h"

#include <locale>
#include <codecvt>
#include <string>

namespace pathtracex {

	class StringUtil {
	public:
		static inline std::wstring bstows(const std::string& bstr) noexcept {
			return conv.from_bytes(bstr);
		}

		static inline std::string wstobs(const std::wstring& wstr) noexcept {
			return conv.to_bytes(wstr);
		}

	private:
		static inline std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
	};

}