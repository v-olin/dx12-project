#pragma once

#include "Event.h"

#include <sstream>

namespace pathtracex {
	
	class KeyEvent : public Event {
	public:
		typedef unsigned char KeyCode;
		unsigned char getKeyCode() const { return keycode; }

		EVENT_CLASS_CATEGORY(EventCategoryKeyboard)

	protected:
		KeyEvent(const KeyCode keycode) : keycode(keycode) { }

		KeyCode keycode;
	};

	class KeyPressedEvent : public KeyEvent {
	public:
		KeyPressedEvent(const KeyCode keycode, bool isRepeat = false) :
			KeyEvent(keycode), isRepeatedEvent(isRepeat) { }

		bool isRepeat() const { return isRepeatedEvent; }

		std::string toString() const override {
			std::stringstream ss;
			ss << "KeyPressedEvent: " << keycode << " (repeated = " << isRepeatedEvent << ")";
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyPressed)

	private:
		bool isRepeatedEvent;
	};

	class KeyReleasedEvent : public KeyEvent {
	public:
		KeyReleasedEvent(const KeyCode keycode) : KeyEvent(keycode) { }

		std::string toString() const override {
			std::stringstream ss;
			ss << "KeyReleasedEvent: " << keycode;
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyReleased)
	};

	class KeyTypedEvent : public KeyEvent {
	public:
		KeyTypedEvent(const KeyCode keycode) : KeyEvent(keycode) { }

		std::string toString() const override {
			std::stringstream ss;
			ss << "KeyTypedEvent: " << keycode;
			return ss.str();
		}

		EVENT_CLASS_TYPE(KeyTyped)
	};

}