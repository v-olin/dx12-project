#pragma once

#include "Window.h"
#include "GUI.h"
#include "DXRenderer.h"

#include <memory>

namespace pathtracex {

	class App {
	public:
		App();
		
		int run();

	private:
		void everyFrame();
		void cleanup();

		Scene scene{};
		GUI gui{scene};
		Window window;
		DXRenderer renderer{window};
	};

}