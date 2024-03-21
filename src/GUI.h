#pragma once

#include "imgui.h"
#include "Window.h"
#include "Scene.h"

namespace pathtracex {

	class GUI {
	public:
		GUI(Scene& scene);
		~GUI();

		void drawGUI();



		Window* window = nullptr;
	private:
		void drawTopMenu();
		void drawModelSelectionMenu();
		void drawSettingsMenu();
		void drawGizmos();

		ImGuiContext* context;

		Scene& scene;
	};

}