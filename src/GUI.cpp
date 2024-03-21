#include "GUI.h"
#include "imgui.h"

#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

namespace pathtracex {

	GUI::GUI() :
		context(nullptr)
	{
		IMGUI_CHECKVERSION();
		context = ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable kbd controls
		ImGui::StyleColorsDark();
	}

	GUI::~GUI() {
		ImGui::DestroyContext();
	}

	void GUI::drawGUI()
	{
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		static bool showGui = true;
		ImGui::ShowDemoWindow(&showGui);

		ImGui::Begin("Path-traceX");
		ImGui::Text("Hello, world!");
		ImGui::End();

		ImGui::Render();
	}
}