#pragma once

#include "imgui.h"
#include "Window.h"
#include "Scene.h"

namespace pathtracex {

	class GUI {
	public:
		GUI(Scene& scene);
		~GUI();

		void drawGUI(RenderSettings& renderSettings);
		void resetContext();

		Window* window = nullptr;
	private:
		void drawTopMenu();
		void drawModelSelectionMenu();
		void drawRightWindow(RenderSettings& renderSettings);
		void drawGizmos(RenderSettings& renderSettings);
		void drawRenderingSettings(RenderSettings& renderSettings);
		void drawSelectableSettings();
		void drawSelectedModelSettings();
		void drawViewport(RenderSettings& renderSettings);

		ImGuiContext* context;

		Scene& scene;

		// The selected object that is displayed in the right window
		std::weak_ptr<Selectable> selectedSelectable;
	};

}