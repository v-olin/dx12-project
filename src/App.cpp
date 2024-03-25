#include "App.h"

#include "PathWin.h"
#include "Logger.h"
#include "EventCallback.h"

namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
		gui.window = &window;

		InputHandler::configure(&window.kbd);

		EventCallback<App> act{ this, dummy };
		//InputHandler::addListener(Keyboard::Event::Press, 'p', act);

		// Initialize renderer
		defaultRenderSettings.camera.transform.setPosition({ 1, 0, -4 });
	}

	void App::dummy() {
		LOG_INFO("exec dummy");
	}

	int App::run() {
		renderer = DXRenderer::getInstance();
		if (!renderer->init(&window))
		{
			MessageBox(0, "Failed to initialize direct3d 12",
				"Error", MB_OK);
			cleanup();
			return 1;
		}

		std::shared_ptr<Model> cube = Model::createPrimative(PrimitiveModelType::CUBE);
		cube->trans.setPosition({ 0, 1, 0 });
		scene.models.push_back(cube);

		std::shared_ptr<Model> cube2 = Model::createPrimative(PrimitiveModelType::PLANE);
		cube2->trans.setPosition({ 1, -1, 0 });
		scene.models.push_back(cube2);


		while(true) {
			const auto ecode = Window::processMessages();
			if (ecode) {
				return *ecode;
			}

			InputHandler::getInstance().processEvents();

			everyFrame();
		}


	}

	void App::cleanup() {

	}

	void App::everyFrame() {
		gui.drawGUI(defaultRenderSettings);
	//	window.pRenderer->onUpdate();
	//	window.pRenderer->onRender();

		// Update render settings
		int width, height;
		window.getSize(width, height);
		defaultRenderSettings.width = width;
		defaultRenderSettings.height = height;

		renderer->Render(defaultRenderSettings, scene);
	}
}