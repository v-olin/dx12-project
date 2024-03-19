#pragma once

#include "Window.h"
#include "ImguiManager.h"

#include <memory>

namespace pathtracex {

	class App {
	public:
		App();
		
		int run();

	private:
		void everyFrame();
		void drawGui();

		Window window;
	};

}