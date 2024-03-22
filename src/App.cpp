#include "App.h"


namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
		gui.window = &window;
		std::shared_ptr<Model> testModel = std::make_shared<Model>();
		scene.models.push_back(testModel);
		std::shared_ptr<Light> testLight = std::make_shared<Light>();
		scene.lights.push_back(testLight);
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

	void App::everyFrame() {
		gui.drawGUI();
		window.pRenderer->onUpdate();
		window.pRenderer->onRender();
	}

	void App::drawGui() {


	}
}