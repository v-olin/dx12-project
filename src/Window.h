#pragma once

#include "Event.h"
#include "PathWin.h"
#include "Keyboard.h"
#include "Mouse.h"
#include "GUI.h"
#include "WindowEvent.h"

#include <chrono>
#include <functional>
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
		HWND windowHandle;
		void getSize(int& width, int& height) const;
		void setInitialized();

		bool windowHasBeenResized();
		void windowResizeEventHandled();
		std::pair<UINT, UINT> getNewWindowSize();

		void updateWindowSize() {
			width = localResizeEvent.w;
			height = localResizeEvent.h;
			localResizeEvent.handled = true;
		}

	private:
		int width, height;
		std::string title;
		bool thisIsInitialized = false;

		struct LocalResizeEvent {
			std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
			UINT w = 0, h = 0;
			bool handled = true;
		};

		LocalResizeEvent localResizeEvent;

		WindowResizeEvent windowResizeEvent{ 0,0 };
		WindowCloseEvent windowCloseEvent{};

		HINSTANCE _hInstance; // i tink program instance?
		static constexpr LPCSTR _wndClassName = "pathtracex::window";

		void registerClass() const noexcept;
		void configureWindow();

		static LRESULT CALLBACK handleMessageSetup(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
		static LRESULT CALLBACK handleMessageThunk(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
		LRESULT handleMessage(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
	};
}