#pragma once

#include "Window.h"
#include "GUI.h"
#include "DXRenderer.h"
#include "RenderSettings.h"
#include "Event.h"
#include "ProcedualWorldManager.h"

#include <memory>
#include <optional>
#include <vector>

namespace pathtracex {
	using EventCallbackFn = std::function<void(Event&)>;

	class AppConfig : public Serializable {
	public:
		//int initialWindowWidth = 1280;
		//int initialWindowHeight = 720;
		int initialWindowWidth = 3440;
		int initialWindowHeight = 1440;
		std::string startupSceneName = "Default";

		std::vector<SerializableVariable> getSerializableVariables() override {
			return {
				{ SerializableType::INT, "Initial Window Width", "The initial width of the window", &initialWindowWidth },
				{ SerializableType::INT, "Initial Window Height", "The initial height of the window", &initialWindowHeight },
				{ SerializableType::STRING, "Startup Scene Name", "The name of the scene that will be loaded on startup", &startupSceneName }
			};
		}
	};

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

		AppConfig config;

		Scene scene{};
		GUI gui{scene};
		Window window;
		DXRenderer* renderer = nullptr;
		Camera defaultCamera{};
		RenderSettings defaultRenderSettings{ 0, 0, defaultCamera };

		bool running = true;

		std::vector<IEventListener*> listeners{};
		std::vector<Event> timedEvents{};

		ProcedualWorldManager worldManager{ {} };
	};

}