#pragma once

#include "imgui.h"

namespace pathtracex {

	class GUI {
	public:
		GUI();
		~GUI();

		void drawGUI();

	private:
		ImGuiContext* context;
	};

}