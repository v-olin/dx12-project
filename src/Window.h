#pragma once

#include "PathWin.h"
#include "Keyboard.h"
#include "Mouse.h"
#include "GUI.h"

#include <memory>
#include <optional>
#include <string>

namespace pathtracex {
	using EventCallbackFn = std::function<void(Event&)>;
	
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
		void setCallbackFn(EventCallbackFn callback);
		static void raiseEvent(Event& e);
	private:
		int width, height;
		std::string title;

		static EventCallbackFn callbackFunc;

		HINSTANCE _hInstance; // i tink program instance?
		static constexpr LPCSTR _wndClassName = "pathtracex::window";

		void registerClass() const noexcept;
		void configureWindow();

		static LRESULT CALLBACK handleMessageSetup(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
		static LRESULT CALLBACK handleMessageThunk(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
		LRESULT handleMessage(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept;
	};
}