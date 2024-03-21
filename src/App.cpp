#include "App.h"


namespace pathtracex {

	App::App() : window(1280, 760, "Path-traceX") {}

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