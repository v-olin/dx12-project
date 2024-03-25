#pragma once

namespace pathtracex {
	class IEventCallback {
	public:
		// invoke callback
		virtual void operator() () = 0;
		// compare callbacks
		virtual bool operator == (IEventCallback* other) = 0;
	};

	template<typename T>
	class EventCallback : public IEventCallback
	{
	public:
		EventCallback(T* instance, void (T::* function)())
			: instance(instance), function(function) {}
		virtual void operator () () override { (instance->*function)(); }

		virtual bool operator == (IEventCallback* other) override
		{
			EventCallback* otherEventCallback = dynamic_cast<EventCallback>(other);
			if (otherEventCallback == nullptr)
				return false;

			return 	(this->function == otherEventCallback->function) &&
				(this->instance == otherEventCallback->instance);
		}
	private:
		// hold instance for callback
		T* instance;
		// hold member function for callback
		void (T::* function)();
	};
}