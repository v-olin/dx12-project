#pragma once

#include "Window.h"
#include "GUI.h"
#include "DXRenderer.h"
#include "RenderSettings.h"
#include "Event.h"

#include <memory>
#include <vector>

namespace pathtracex {

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

	private:
		App();
		void everyFrame();
		void cleanup();
		void onEvent(Event& e);

		Scene scene{};
		GUI gui{scene};
		Window window;
		DXRenderer* renderer = nullptr;
		Camera defaultCamera{};
		RenderSettings defaultRenderSettings{ 0, 0, defaultCamera };

		std::vector<IEventListener*> listeners{};
	};

}