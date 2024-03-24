#include "App.h"
#include <Windows.h>

namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
		gui.window = &window;

		// Initialize renderer
		defaultRenderSettings.camera.transform.setPosition({ 1, 0, -4 });
	}

	int App::run() {
		if (!renderer.InitD3D())
		{
			MessageBox(0, "Failed to initialize direct3d 12",
				"Error", MB_OK);
			cleanup();
			return 1;
		}


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
		gui.drawGUI(defaultRenderSettings);
	//	window.pRenderer->onUpdate();
	//	window.pRenderer->onRender();

		// Update render settings
		int width, height;
		window.getSize(width, height);
		defaultRenderSettings.width = width;
		defaultRenderSettings.height = height;

		renderer.Render(defaultRenderSettings);
	}
}