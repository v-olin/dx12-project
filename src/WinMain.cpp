#include <stdlib.h>
#include <string>
#include <sstream>
#include <spdlog/spdlog.h>

#include "PathWin.h"
#include "App.h"
#include "Logger.h"
#include "Exceptions.h"
#include "Window.h"

LPCWSTR towstr(const char* str) {
	size_t len = strlen(str);
	wchar_t* wstr = static_cast<wchar_t*>(malloc(sizeof(wchar_t) * len));
	size_t chars = 0;
	mbstowcs_s(&chars, wstr, len / 4, str, len);
	LPCWSTR pwstr = wstr;
	return pwstr;
}

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	try {
		AllocConsole();
		pathtracex::Logger::init();
		return pathtracex::App{}.run();
	}
	catch (const pathtracex::HRException ex) {
		MessageBox(nullptr, ex.what(), ex.getType(), MB_OK | MB_ICONEXCLAMATION);
	}
	catch (const pathtracex::PathException ex) {
		MessageBox(nullptr, ex.what(), ex.getType(), MB_OK | MB_ICONEXCLAMATION);
	}
	catch (const std::exception& ex) {
		MessageBox(nullptr, ex.what(), "unknown error", MB_OK | MB_ICONEXCLAMATION);
	}
	return -1;
}