#include "App.h"

#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

namespace pathtracex {

	App::App() :
		window(1280, 760, "Path-traceX")
	{ }

	int App::run() {
		while(true) {
			const auto ecode = Window::processMessages();
			if (ecode) {
				return *ecode;
			}

			everyFrame();
		}


	}

	void App::cleanup() {

	}

	void App::everyFrame() {
		drawGui();
		window.pRenderer->onUpdate();
		window.pRenderer->onRender();
	}

	void App::drawGui() {

		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		static bool showGui = true;
		ImGui::ShowDemoWindow(&showGui);

		ImGui::Render();
	}
}