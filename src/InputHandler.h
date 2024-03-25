#pragma once

#include "EventCallback.h"
#include "Keyboard.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace std {
	template<>
	struct std::hash<pathtracex::Keyboard::Event> {
		std::size_t operator()(const pathtracex::Keyboard::Event& ev) const {
			int itype = static_cast<int>(ev.type);

			return (static_cast<size_t>(itype) << 16) | ev.code;
		}
	};
}


namespace pathtracex {
	typedef std::shared_ptr<IEventCallback> Action;
	using KeyboardMap = std::unordered_map<Keyboard::Event, std::vector<Action>, std::hash<Keyboard::Event>>;

	class InputHandler {
	public:
		static InputHandler& getInstance() {
			static InputHandler instance;
			return instance;
		}

		InputHandler(InputHandler const&) = delete;
		void operator=(InputHandler const&) = delete;

		static void configure(Keyboard* kbd);
		void processEvents();
		void addListener(Keyboard::Event event, Action callback);
		void addListener(Keyboard::Event::Type type, unsigned char keycode, Action callback);

	private:
		InputHandler() : keyboard(nullptr) {}

		Keyboard* keyboard;
		KeyboardMap mappings{};

	};

}