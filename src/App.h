#pragma once

#include "Window.h"
#include "GUI.h"
#include "DXRenderer.h"
#include "RenderSettings.h"
#include "InputHandler.h"

#include <memory>

namespace pathtracex {

	class App {
	public:
		App();
		
		int run();

	private:
		void everyFrame();
		void cleanup();

		void dummy();

		Scene scene{};
		GUI gui{scene};
		Window window;
		DXRenderer* renderer = nullptr;
		Camera defaultCamera{};
		RenderSettings defaultRenderSettings{ 0, 0, defaultCamera };
	};

}