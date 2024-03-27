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

		static inline std::string generateRandomString(int length) noexcept {
			std::string str;
			static const char alphanum[] =
				"0123456789"
				"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				"abcdefghijklmnopqrstuvwxyz";

			for (int i = 0; i < length; ++i) {
				str += alphanum[rand() % (sizeof(alphanum) - 1)];
			}
			return str;
		}
	};
}