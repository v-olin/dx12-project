#pragma once

#include "PathWin.h"

#include <locale>
#include <codecvt>
#include <string>
#include <random>

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

			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);

			for (int i = 0; i < length; ++i) {
				str += alphanum[dis(gen)];
			}

			return str;
		}
	};
}