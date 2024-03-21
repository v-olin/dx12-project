#include "GUI.h"
#include "imgui.h"

#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

namespace pathtracex {

#define SHOW_DEMO_WINDOW false

	GUI::GUI(Scene& scene) : scene(scene)
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

		drawTopMenu();
		drawModelSelectionMenu();
		drawSettingsMenu();

#if SHOW_DEMO_WINDOW
		ImGui::ShowDemoWindow();
#endif


		ImGui::Render();
	}
	void GUI::drawModelSelectionMenu()
	{
		int w, h;
		window->getSize(w, h);
		int panelWidth = w / 5;
		ImGui::SetNextWindowPos(ImVec2(0, 18));
		ImGui::SetNextWindowSize(ImVec2(panelWidth, h));

		ImGuiWindowFlags windowFlags = 0;
		windowFlags |= ImGuiWindowFlags_NoTitleBar;
		windowFlags |= ImGuiWindowFlags_NoMove;
		windowFlags |= ImGuiWindowFlags_NoScrollbar;

		ImGui::Begin("Model selection", nullptr, windowFlags);


		for (auto model : scene.models)
		{
			ImGui::Selectable(model->name.c_str());
		}

		ImGui::End();
	}

	void GUI::drawSettingsMenu()
	{
		int w, h;
		window->getSize(w, h);
		int panelWidth = w / 5;
		ImGui::SetNextWindowPos(ImVec2(w - panelWidth, 18));
		ImGui::SetNextWindowSize(ImVec2(panelWidth, h));

		ImGuiWindowFlags windowFlags = 0;
		windowFlags |= ImGuiWindowFlags_NoTitleBar;
		windowFlags |= ImGuiWindowFlags_NoMove;
		windowFlags |= ImGuiWindowFlags_NoScrollbar;

		ImGui::Begin("Settings", nullptr, windowFlags);
		ImGui::Text("Settings");
		ImGui::End();
	}

	void GUI::drawGizmos()
	{

	}

	void GUI::drawTopMenu()
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("File"))
		{

		}

		ImGui::EndMainMenuBar();
	}
}