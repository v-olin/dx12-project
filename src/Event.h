#pragma once

#include <string>

#define BIT(x)	(1 << x)

// i am not responsible for this
#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }

namespace pathtracex {
	enum class EventType { 
		None = 0,
		KeyPressed, 
		KeyReleased, 
		KeyTyped,
		MouseMoved,
		MousePressed,
		MouseReleased,
		MouseScrolled
	};

	enum EventCategory {
		None = 0,
		EventCategoryApp = BIT(0),
		EventCategoryKeyboard = BIT(1),
		EventCategoryMouse = BIT(2)
	};

#define EVENT_CLASS_TYPE(type) static EventType getStaticType() { return EventType::type; }\
								virtual EventType getEventType() const override { return getStaticType(); }\
								virtual std::string getName() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) virtual int getCategoryFlags() const override { return category; }

	class Event	{
	public:
		virtual ~Event() = default;

		bool Handled = false;

		virtual EventType getEventType() const = 0;
		virtual std::string getName() const = 0;
		virtual int getCategoryFlags() const = 0;
		virtual std::string toString() const { return getName(); }

		bool isInCategory(EventCategory category) {
			return getCategoryFlags() & category;
		}
	};

	class EventDispatcher {
	public:
		EventDispatcher(Event& event) : event(event) { }

		// compiler will deduce F
		template<typename T, typename F>
		bool dispatch(const F& func) {
			if (event.getEventType() == T::getStaticType()) {
				event.Handled |= func(static_cast<T&>(event));
			}
		}

	private:
		Event& event;
	};

	class IEventListener {
	public:
		virtual void onEvent(Event& e) = 0;
	};
}