#pragma once

#include "Keyboard.h"

namespace pathtracex {

	class InputHandler {
		friend class Keyboard;
	public:
		InputHandler(Keyboard& kbd);
		~InputHandler() = default;

		void processEvents();

	private:
		Keyboard& keyboard;

	};

}