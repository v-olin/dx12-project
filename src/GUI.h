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
		void drawRightWindow();
		void drawGizmos();
		void drawRenderingSettings();
		void drawSelectableSettings();
		void drawSelectedModelSettings();

		ImGuiContext* context;

		Scene& scene;

		// The selected object that is displayed in the right window
		std::weak_ptr<Selectable> selectedSelectable;
	};

}