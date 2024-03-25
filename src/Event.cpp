#include "Event.h"

#include "Logger.h"

namespace pathtracex {

	void Event::addListener(IEventCallback* action) {
		auto position = find(actions.begin(), actions.end(), action);

		if (position != actions.end()) {
			LOG_WARN("Action existed in delegate list.");
			return;
		}

		actions.push_back(action);
	}

	void Event::removeListener(IEventCallback* action) {
		auto position = find(actions.begin(), actions.end(), action);

		if (position == actions.end()) {
			return;
		}

		actions.erase(position);
	}

	void Event::fire() {
		for (IEventCallback* action : actions) {
			(*action)();
		}
	}
}