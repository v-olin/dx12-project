#include <stdlib.h>
#include <string>
#include <sstream>

#include "PathWin.h"
#include "App.h"
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
		return pathtracex::App{}.run();
	}
	catch (const std::exception& ex) {
		MessageBox(nullptr, ex.what(), "skill issue", MB_OK | MB_ICONEXCLAMATION);
	}
	return -1;
}