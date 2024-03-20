#pragma once

#include "PathWin.h"
#include "Keyboard.h"
#include "Mouse.h"
#include "DXRenderer.h"
#include "ImguiManager.h"

#include <memory>
#include <optional>
#include <string>

namespace pathtracex {
	class Window {
	public:
		Window(int width, int height, const std::string& title) noexcept;
		~Window();
		Window(const Window&) = delete;
		Window& operator=(const Window&) = delete;

		void setTitle(const std::string& title);
		static std::optional<int> processMessages();

		Keyboard kbd;
		Mouse mouse;
		// defer construction of renderer since it needs the
		// window handle and the imgui win32 wndproc hook
		std::unique_ptr<DXRenderer> pRenderer;
		std::unique_ptr<ImguiManager> pImgui;

	private:
		int width, height;
		std::string title;
		HWND windowHandle;

		HINSTANCE _hInstance; // i tink program instance?
		static constexpr LPCSTR _wndClassName = "pathtracex::window";

		void registerClass() const noexcept;
		void configureWindow();

		static LRESULT CALLBACK handleMessageSetup(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
		static LRESULT CALLBACK handleMessageThunk(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
		LRESULT handleMessage(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
	};
}