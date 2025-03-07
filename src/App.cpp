#include "App.h"
#include "Serializer.h"
#include "Event.h"
#include "Logger.h"
#include "PathWin.h"
#include "Window.h"
#include "Pong.h"

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
		pongGame.renderSettings.width = width;
		pongGame.renderSettings.height = height;

		defaultRenderSettings.camera.transform.setPosition({ 1, 0, -4 });
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

		std::string gameName{ "pong" };
		if (gameName.compare(config.startupSceneName) == 0) {
			isPlayingGame = true;
		}

		if (isPlayingGame) {
			pongGame.initGame();
			pongGame.renderSettings.raytracingSupported = renderer->raytracingIsSupported();
			renderer->initRaytracingPipeline(pongGame.renderSettings, pongGame.scene);
		}
		else {
			Serializer::deserializeScene(config.startupSceneName, scene);
			scene.procedualWorldManager = &worldManager;
			scene.procedualWorldManager->createMaterial();
			defaultRenderSettings.raytracingSupported = renderer->raytracingIsSupported();
		}

		if (defaultRenderSettings.raytracingSupported) {
			renderer->initRaytracingPipeline(defaultRenderSettings, scene);
		}

		if (!isPlayingGame) {
			registerEventListener(&defaultCamera);
		}
		else {
			registerEventListener(&pongGame);
		}

		while(running) {
			const auto ecode = Window::processMessages();
			if (ecode)
			{
				return *ecode;
			}

			if (!isPlayingGame)
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

	void App::cleanup() { }

	void App::everyFrame() {
		if (isPlayingGame) {
			int width, height;
			window.getSize(width, height);
			pongGame.renderSettings.width = width;
			pongGame.renderSettings.height = height;

			pongGame.everyFrame();

			renderer->Render(pongGame.renderSettings, pongGame.scene);
		}
		else {
			if (defaultRenderSettings.drawProcedualWorld)
			{
				worldManager.updateProcedualWorld(defaultCamera);
			}

			scene.proceduralGroundModels = worldManager.procedualWorldGroundModels;
			scene.proceduralTreeModels = worldManager.procedualWorldTreeModels;
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
}