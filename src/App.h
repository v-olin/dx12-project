#pragma once

#include "Window.h"
#include "GUI.h"

#include <memory>

namespace pathtracex {

	class App {
	public:
		App();
		
		int run();

	private:
		void everyFrame();
		void drawGui();

		GUI gui{};
		Window window;
	};

}