#include "App.h"
#include <Windows.h>

namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
	//	gui.window = &window;

		// Initialize renderer
	

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
		//gui.drawGUI();
	//	window.pRenderer->onUpdate();
	//	window.pRenderer->onRender();
	}

	void App::drawGui() {


	}
}