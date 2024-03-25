#include "InputHandler.h"

#include "Logger.h"

namespace pathtracex {

	void InputHandler::configure(Keyboard* kbd) {
		InputHandler& inst = getInstance();
		inst.keyboard = kbd;
	}
	
	void InputHandler::processEvents() {
		std::optional<Keyboard::Event> ev;

		while ((ev = keyboard->readKey()).has_value()) {
			auto position = mappings.find(ev.value());

			if (position == mappings.end()) {
				continue;
			}

			for (const auto callback : mappings[ev.value()]) {
				(*callback)();
			}
		}
	}

	void InputHandler::addListener(Keyboard::Event event, Action callback) {
		if (mappings.find(event) == mappings.end()) {
			mappings[event] = { callback };
		}
		else {
			mappings[event].push_back(callback);
		}
	}

	void InputHandler::addListener(Keyboard::Event::Type type, unsigned char keycode, Action callback) {
		addListener({ type, keycode }, callback);
	}
}