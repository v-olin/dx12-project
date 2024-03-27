#pragma once

#include "Event.h"

#include "PathWin.h"
#include <sstream>

namespace pathtracex {
	
	class WindowResizeEvent : public Event {
	public:
		WindowResizeEvent(UINT width, UINT height) :
			width(width), height(height) { }
	
		UINT getWidth() const { return width; }
		UINT getHeight() const { return height; }

		std::string toString() const override {
			std::stringstream ss;
			ss << "WindowResizeEvent: (" << width << ", " << height << ")";
			return ss.str();
		}

		EVENT_CLASS_TYPE(WindowResize)
		EVENT_CLASS_CATEGORY(EventCategoryWindow)

	private:
		UINT width, height;
	};

	class WindowCloseEvent : public Event {
	public:
		WindowCloseEvent() = default;

		EVENT_CLASS_TYPE(WindowClose)
		EVENT_CLASS_CATEGORY(EventCategoryWindow)
	};

}