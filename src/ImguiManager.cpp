#include "ImguiManager.h"
#include "imgui.h"

namespace pathtracex {

	ImguiManager::ImguiManager() :
		context(nullptr)
	{
		IMGUI_CHECKVERSION();
		context = ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable kbd controls
		ImGui::StyleColorsDark();
	}

	ImguiManager::~ImguiManager() {
		ImGui::DestroyContext();
	}

}