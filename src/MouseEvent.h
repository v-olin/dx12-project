#pragma once

#include "Event.h"

#include <sstream>

namespace pathtracex {
	class MouseMovedEvent : public Event {
	public:
		MouseMovedEvent(const float x, const float y) :
			posx(x), posy(y) { }

		float getX() const { return posx; }
		float getY() const { return posy; }

		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseMovedEvent: (" << posx << ", " << posy << ")";
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseMoved)
		EVENT_CLASS_CATEGORY(EventCategoryMouse)

	private:
		float posx, posy;
	};

	enum MouseButtonType {
		LeftButton = 0,
		MiddleButton,
		RightButton
	};

	class MouseButtonEvent : public Event {
	public:
		MouseButtonType getMouseButton() const { return button; }

		EVENT_CLASS_CATEGORY(EventCategoryMouse)
	protected:
		MouseButtonEvent(const MouseButtonType button)
			: button(button) { }

		MouseButtonType button;
	};

	class MouseButtonPressedEvent : public MouseButtonEvent {
	public:
		MouseButtonPressedEvent(const MouseButtonType button)
			: MouseButtonEvent(button) { }

		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseButtonPressedEvent: " << button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MousePressed)
	};

	class MouseButtonReleasedEvent : public MouseButtonEvent {
	public:
		MouseButtonReleasedEvent(const MouseButtonType button)
			: MouseButtonEvent(button) { }

		std::string toString() const override {
			std::stringstream ss;
			ss << "MouseButtonReleasedEvent: " << button;
			return ss.str();
		}

		EVENT_CLASS_TYPE(MouseReleased)
	};
}