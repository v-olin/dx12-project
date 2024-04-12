#include "App.h"
#include "Serializer.h"
#include "Event.h"
#include "Logger.h"
#include "PathWin.h"
#include "Window.h"

namespace pathtracex
{
	App::App() : config(Serializer::deserializeConfig()), window(config.initialWindowWidth, config.initialWindowHeight, "PathTracer")
	{
		gui.window = &window;
		window.setInitialized();

		callback = BIND_EVENT_FN(App::onEvent);

		// Initialize renderer
		int width, height;
		window.getSize(width, height);
		defaultRenderSettings.width = width;
		defaultRenderSettings.height = height;

		defaultRenderSettings.camera.transform.rotate({ 0, 0, 1 }, 1.571);
		defaultRenderSettings.camera.transform.setPosition({ 0, 100, 0 });
		//defaultRenderSettings.camera.transform.transformMatrix = DirectX::XMMatrixLookAtRH(defaultRenderSettings.camera.transform.getPosition(), DirectX::XMVectorSet(0, 0, 0, 0), DirectX::XMVectorSet(0, 1, 0, 0));
	}

	int App::run()
	{
		renderer = DXRenderer::getInstance();
		if (!renderer->init(&window))
		{
			MessageBox(0, "Failed to initialize direct3d 12",
					   "Error", MB_OK);
			cleanup();
			return 1;
		}


		registerEventListener(&defaultCamera);

		Serializer::deserializeScene(config.startupSceneName, scene);

		scene.procedualWorldManager = &worldManager;

		while(running) {
			const auto ecode = Window::processMessages();
			if (ecode)
			{
				return *ecode;
			}

			defaultCamera.updateMovement();
			everyFrame();
		}

		Serializer::serializeScene(scene);
		Serializer::serializeConfig(config);

		cleanup();
	}

	void App::registerEventListener(IEventListener *listener)
	{
		getInstance().listeners.push_back(listener);
	}

	void App::raiseEvent(Event& e) {
		App& inst = getInstance();

		if (!inst.callback.has_value()) [[unlikely]] {
			LOG_ERROR("Event callback handler not set!! Very bad!!");
			return;
		}

		(*inst.callback)(e);
	}

	/*
	void App::raiseTimeEvent(Event& e) {
		// this will make a copy but that's fine
		getInstance().timedEvents.push_back(e);
	}

	void App::raiseTimedEvents() {
		for (auto timedEvent : timedEvents) {
			// if timedEvent.shouldFire()
			//		raise(timedEvent)
		}
	}
	*/

	// TODO: This should be in a separate class
	void App::onEvent(Event &e)
	{
		EventDispatcher dispatcher(e);

		for (auto listener : listeners)
		{
			if (e.Handled)
			{
				break;
			}

			listener->onEvent(e);
		}

		if (e.getEventType() == EventType::WindowClose)
		{
			running = false;
		}

		e.Handled = true;
	}

	void App::cleanup()
	{
	}

	void App::everyFrame() {
		if (defaultRenderSettings.drawProcedualWorld)
		{
			worldManager.updateProcedualWorld(defaultCamera);
		}

		scene.proceduralGroundModels = worldManager.procedualWorldGroundModels;
		scene.proceduralSkyModels = worldManager.procedualWorldSkyModels;

		if (window.windowHasBeenResized()) {
			auto newSize = window.getNewWindowSize();
			WindowResizeEvent wre{ newSize.first, newSize.second };
			window.updateWindowSize();
			App::raiseEvent(wre);
			//gui.resetContext();
		}


		// Update render settings
		int width, height;
		window.getSize(width, height);
		defaultRenderSettings.width = width;
		defaultRenderSettings.height = height;

		gui.drawGUI(defaultRenderSettings);

		renderer->Render(defaultRenderSettings, scene);

	}
}