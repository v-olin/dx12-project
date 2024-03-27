#pragma once

#include "Window.h"
#include "GUI.h"
#include "DXRenderer.h"
#include "RenderSettings.h"
#include "Event.h"

#include <memory>
#include <optional>
#include <vector>

namespace pathtracex {
	using EventCallbackFn = std::function<void(Event&)>;

	class App {
	public:
		static App& getInstance() {
			static App instance;
			return instance;
		}

		App(App const&) = delete;
		void operator=(App const&) = delete;

		int run();
		static void registerEventListener(IEventListener* listener);
		static void raiseEvent(Event& e);
		static void raiseTimeEvent(Event& e);

	private:
		App();
		void everyFrame();
		void cleanup();
		void onEvent(Event& e);
		void raiseTimedEvents();
		std::optional<EventCallbackFn> callback;

		Scene scene{};
		GUI gui{scene};
		Window window;
		DXRenderer* renderer = nullptr;
		Camera defaultCamera{};
		RenderSettings defaultRenderSettings{ 0, 0, defaultCamera };

		std::vector<IEventListener*> listeners{};
		std::vector<Event> timedEvents{};
	};

}