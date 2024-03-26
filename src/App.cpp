#include "App.h"

#include "Event.h"
#include "Logger.h"
#include "PathWin.h"
#include "Window.h"

namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
		gui.window = &window;

		callback = BIND_EVENT_FN(App::onEvent);

		// Initialize renderer
		defaultRenderSettings.camera.transform.setPosition({ 1, 0, -4 });
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

			everyFrame();
		}
	}

	void App::registerEventListener(IEventListener* listener) {
		getInstance().listeners.push_back(listener);
	}

	void App::raiseEvent(Event& e) {
		getInstance().callback(e);
	}

	void App::onEvent(Event& e) {
		EventDispatcher dispatcher(e);

		for (auto listener : listeners) {
			if (e.Handled) {
				break;
			}

			listener->onEvent(e);
		}

		e.Handled = true;
	}

	void App::cleanup() {

	}

	void App::everyFrame() {
		gui.drawGUI(defaultRenderSettings);

		// Update render settings
		int width, height;
		window.getSize(width, height);
		defaultRenderSettings.width = width;
		defaultRenderSettings.height = height;

		renderer->Render(defaultRenderSettings, scene);
	}
}