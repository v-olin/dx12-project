#include "App.h"


namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
	//	gui.window = &window;

		// Initialize renderer
	

	}

	int App::run() {
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