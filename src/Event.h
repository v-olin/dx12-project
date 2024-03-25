#pragma once

#include "EventCallback.h"

#include <vector>

namespace pathtracex {
	class Event	{
	public:
		Event() = default;
		~Event() = default;

		void addListener(IEventCallback* action);
		void removeListener(IEventCallback* action);
		void fire();

	private:
		std::vector<IEventCallback*> actions;
	};
}