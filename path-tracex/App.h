#pragma once

#include "Window.h"

namespace pathtracex {

	class App {
	public:
		App();
		
		int run();

	private:
		Window window;
	};

}