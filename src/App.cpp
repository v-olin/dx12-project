#include "App.h"
#include <Windows.h>

namespace pathtracex {
	App::App() : window(1280, 720, "PathTracer")
	{
		gui.window = &window;

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

		//std::shared_ptr<Model> cube = Model::createPrimative(PrimitiveModelType::CUBE);
		//cube->trans.setPosition({ 0, 1, 0 });
		//scene.models.push_back(cube);

		//std::shared_ptr<Model> plane = Model::createPrimative(PrimitiveModelType::PLANE);
		//plane->trans.setPosition({ 1, -1, 0 });
		//scene.models.push_back(plane);

		//std::shared_ptr<Model> sphere = Model::createPrimative(PrimitiveModelType::SPHERE);
		//sphere->trans.setPosition({ 2.5, 1, 0 });
		//scene.models.push_back(sphere);


		std::shared_ptr<Model> space_ship = std::make_shared<Model>("../../assets/chopper.obj");
		space_ship->trans.setPosition({ 1, 1, 5 });
		scene.models.push_back(space_ship);



		while(true) {
			const auto ecode = Window::processMessages();
			if (ecode) {
				return *ecode;
			}

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