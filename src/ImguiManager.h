#pragma once

#include "imgui.h"

namespace pathtracex {

	class ImguiManager {
	public:
		ImguiManager();
		~ImguiManager();

	private:
		ImGuiContext* context;
	};

}