#include "InputHandler.h"

#include "Logger.h"

namespace pathtracex {

	InputHandler::InputHandler(Keyboard& kbd) :
		keyboard(kbd)
	{

	}
	
	void InputHandler::processEvents() {
		std::optional<Keyboard::Event> ev;
		
		while ((ev = keyboard.readKey()).has_value()) {

		}
	}
}