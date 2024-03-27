#pragma once

#include "Event.h"

#include <chrono>
#include <sstream>

namespace pathtracex {
	
	/* this does not work yet, do not use!! */
	class TimedEvent : public Event {
		using clock = std::chrono::steady_clock;
	public:
		TimedEvent(clock::duration interval, bool isPeriodic = false) :
			eventStart(std::chrono::steady_clock::now()),
			eventInterval(interval),
			isPeriodic(isPeriodic) { }

		bool shouldFire() {
			bool timeHasPassed = eventStart + eventInterval > clock::now();

			if (timeHasPassed && isPeriodic) {
				eventStart = clock::now();
			}

			return timeHasPassed;
		}

		EVENT_CLASS_CATEGORY(EventCategoryTime)
		EVENT_CLASS_TYPE(TimeElapsed)

	protected:
		bool isPeriodic;
		clock::time_point eventStart;
		clock::duration eventInterval;
	};

}