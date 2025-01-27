#include "Window.h"

#include "StringUtil.h"
#include "App.h"
#include "WindowEvent.h"
#include "backends/imgui_impl_win32.h"
#include "Logger.h"

#define WINDOW_STYLING (WS_CAPTION | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU | WS_THICKFRAME)

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace pathtracex {

	Window::Window(int width, int height, const std::string& title) noexcept
		: width(width)
		, height(height)
		, title(title)
		, _hInstance(GetModuleHandle(nullptr))
		, windowHandle(nullptr)
		
	{
		//kbd.enableAutorepeat();
		LOG_INFO("Creating window");
		registerClass();
		configureWindow();
		LOG_INFO("Window created");
	}
	
	Window::~Window() {
		ImGui_ImplWin32_Shutdown();
		DestroyWindow(windowHandle);
		UnregisterClass(_wndClassName, _hInstance);
	}

	void Window::setTitle(const std::string& newTitle) {
		title = newTitle;
		if (SetWindowText(windowHandle, newTitle.c_str()) == 0) {
			// throw except
		}
	}

	std::optional<int> Window::processMessages() {
		MSG msg;

		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				return msg.wParam;
			}

			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		return {};
	}

	void Window::getSize(int& width, int& height) const
	{
		width = this->width;
		height = this->height;
	}

	void Window::registerClass() const noexcept {
		WNDCLASSEX wc{ 0 };
		wc.cbSize = sizeof(wc);

		// give each window of app their own device context
		// they may be rendered independently
		wc.style = CS_OWNDC;

		// pointer to function which handles window messages
		wc.lpfnWndProc = handleMessageSetup;

		// extra memory for window struct
		wc.cbClsExtra = 0;

		// extra memory for each window instance
		wc.cbWndExtra = 0;

		// spawned window
		wc.hInstance = _hInstance;

		// set default windows values
		wc.hIcon = nullptr;
		wc.hCursor = nullptr;
		wc.hbrBackground = nullptr;

		// don't use windows menus, euw!!!
		wc.lpszMenuName = nullptr;

		wc.lpszClassName = _wndClassName;

		// for custom icon
		// TODO K�L: l�gg till cool icon
		wc.hIconSm = nullptr;

		RegisterClassEx(&wc);
	}

	void Window::configureWindow() {
		int off_x = 100;
		int off_y = 100;

		RECT wr{
			off_x,			// LONG left
			off_y,			// LONG top
			off_x + width,	// LONG right
			off_y + height	// LONG bottom
		};

		if ((AdjustWindowRect(&wr, WINDOW_STYLING, FALSE)) == 0) {
			// TODO throw except
		}

		windowHandle = CreateWindow(
			_wndClassName, title.c_str(),
			WINDOW_STYLING,
			CW_USEDEFAULT, CW_USEDEFAULT,
			wr.right - wr.left, wr.bottom - wr.top,
			nullptr, nullptr, _hInstance, this
		);

		if (windowHandle == nullptr) {
			LOG_FATAL("Failed to create window");
			throw std::runtime_error("unluko");
		}
		

		// hook imgui to wndproc before creating renderer
		ImGui_ImplWin32_Init(windowHandle);

		ShowWindow(windowHandle, SW_SHOWDEFAULT);
	}

	LRESULT WINAPI Window::handleMessageSetup(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept {
		// nccreate = window creation message
		if (message != WM_NCCREATE) {
			return DefWindowProc(winHandle, message, wparam, lparam);
		}

		const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lparam);

		// lpCreateParams is the 'this' in CreateWindow in ctor
		Window* const pwnd = static_cast<Window*>(pCreate->lpCreateParams);

		SetWindowLongPtr(winHandle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pwnd));
		SetWindowLongPtr(winHandle, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::handleMessageThunk));

		return pwnd->handleMessage(winHandle, message, wparam, lparam);
	}

	LRESULT WINAPI Window::handleMessageThunk(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept {
		Window* const pwnd = reinterpret_cast<Window*>(GetWindowLongPtr(winHandle, GWLP_USERDATA));

		return pwnd->handleMessage(winHandle, message, wparam, lparam);
	}

	void Window::setInitialized() {
		thisIsInitialized = true;
	}

	bool Window::windowHasBeenResized() {
		auto timeSinceLastResize =
			std::chrono::steady_clock::now() - localResizeEvent.time;
		return !localResizeEvent.handled && (timeSinceLastResize > std::chrono::milliseconds(500));
	}

	void Window::windowResizeEventHandled() {
		localResizeEvent.handled = true;
	}

	std::pair<UINT, UINT> Window::getNewWindowSize() {
		return { localResizeEvent.w, localResizeEvent.h };
	}

	LRESULT Window::handleMessage(HWND winHandle, UINT message, WPARAM wparam, LPARAM lparam) noexcept {
		if (ImGui_ImplWin32_WndProcHandler(winHandle, message, wparam, lparam)) {
			return true;
		}
		
		switch (message) {
			/* If user presses close button on window */
		case WM_CLOSE:
			//PostQuitMessage(0); // post quit message
			//return 0; // signal to windows message was handled
			LOG_INFO("WM_CLOSE");
			App::raiseEvent(this->windowCloseEvent);
			return 0;
			/* If window unfocused, clear keyboard */
		case WM_KILLFOCUS:
			LOG_INFO("WM_KILLFOCUS");
			kbd.clearState();
			break;
		case WM_SIZE: {
			LOG_INFO("WM_SIZE");
			// this is just to prevent the program from hanging
			// at the start since a WM_SIZE message is posted when
			// the window is created
			if (thisIsInitialized && lparam) [[likely]] {
				localResizeEvent.time = std::chrono::steady_clock::now();
				localResizeEvent.w = LOWORD(lparam);
				localResizeEvent.h = HIWORD(lparam);
				localResizeEvent.handled = false;
			}
			return 0;
			break;
		}
			/* Keyboard messages */
		case WM_KEYDOWN:
		case WM_SYSKEYDOWN:
			LOG_INFO("WM_KEYDOWN");
			if (!(lparam & 0x40000000) || kbd.autorepeatIsEnabled()) {
				kbd.onKeyPressed(static_cast<unsigned char>(wparam));
			}
			break;
		case WM_KEYUP:
		case WM_SYSKEYUP:
			LOG_INFO("WM_KEYUP");
			kbd.onKeyReleased(static_cast<unsigned char>(wparam));
			break;
		case WM_CHAR:
			LOG_INFO("WM_CHAR");
			kbd.onChar(static_cast<unsigned char>(wparam));
			break;

			/* Mouse messages */
		case WM_MOUSEMOVE: {
			const POINTS pt = MAKEPOINTS(lparam);
			// if in client region -> log everything
			if (pt.x >= 0 && pt.x < width && pt.y >= 0 && pt.y < height) {
				mouse.onMouseMove(pt.x, pt.y);
				if (!mouse.isInWindow()) {
					SetCapture(windowHandle);
					mouse.onMouseEnter();
				}
			}
			// if not in client region -> log only if button is pressed
			else {
				LOG_INFO("WM_MOUSEMOVE OUTSIDE BUT BUTTON PRESSED");
				if (wparam & (MK_LBUTTON | MK_RBUTTON)) {
					mouse.onMouseMove(pt.x, pt.y);
				}
				else {
					ReleaseCapture();
					mouse.onMouseLeave();
				}
			}
			break;
		}
		case WM_LBUTTONDOWN: {
			LOG_INFO("WM_LBUTTONDOWN");
			const POINTS pt = MAKEPOINTS(lparam);
			mouse.onLeftPressed(pt.x, pt.y);
			break;
		}
		case WM_RBUTTONDOWN: {
			LOG_INFO("WM_RBUTTONDOWN");
			const POINTS pt = MAKEPOINTS(lparam);
			mouse.onRightPressed(pt.x, pt.y);
			break;
		}
		case WM_LBUTTONUP: {
			LOG_INFO("WM_LBUTTONUP");
			const POINTS pt = MAKEPOINTS(lparam);
			mouse.onLeftReleased(pt.x, pt.y);
			break;
		}
		case WM_RBUTTONUP: {
			LOG_INFO("WM_RBUTTONUP");
			const POINTS pt = MAKEPOINTS(lparam);
			mouse.onRightReleased(pt.x, pt.y);
			break;
		}
		case WM_MOUSEWHEEL: {
			LOG_INFO("WM_MOUSEWHEEL");
			const POINTS pt = MAKEPOINTS(lparam);
			short wheeldelta = GET_WHEEL_DELTA_WPARAM(wparam);
			mouse.onWheelDelta(pt.x, pt.y, wheeldelta);
			break;
		}
		}

		// if other BS-message, let windows handle that shit
		return DefWindowProc(winHandle, message, wparam, lparam);
	}
}