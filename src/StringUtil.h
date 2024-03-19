#pragma once

#include "PathWin.h"

#include <locale>
#include <codecvt>
#include <string>

namespace pathtracex {

	class StringUtil {
	public:
		static inline std::wstring bstows(const std::string& bstr) noexcept {
			std::wstring wstr(bstr.begin(), bstr.end());
			return wstr;
		}

		static inline std::string wstobs(const std::wstring& wstr) noexcept {
			std::string str(wstr.begin(), wstr.end());
			return str;
		}
	};
}