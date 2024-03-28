#include "GUI.h"
#include "imgui.h"

#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"
#include "../../vendor/ImGuizmo/ImGuizmo.h"
#include <DirectXMath.h>
#include <imgui_internal.h>
#include "ResourceManager.h"
#include "Serializer.h"

namespace pathtracex {

#define SHOW_DEMO_WINDOW false

	GUI::GUI(Scene& scene) : scene(scene)
	{
		IMGUI_CHECKVERSION();
		context = ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // enable kbd controls
		ImGui::StyleColorsDark();
	}

	GUI::~GUI() {
		ImGui::DestroyContext();
	}

	void GUI::resetContext() {

	}

	void GUI::drawGUI(RenderSettings& renderSettings)
	{
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();

		drawTopMenu();
		drawModelSelectionMenu();
		drawRightWindow(renderSettings);
		drawViewport(renderSettings);

#if SHOW_DEMO_WINDOW
		ImGui::ShowDemoWindow();
#endif


		ImGui::Render();
	}
	void GUI::drawModelSelectionMenu()
	{
		int w, h;
		window->getSize(w, h);
		int panelWidth = w / 5;
		ImGui::SetNextWindowPos(ImVec2(0, 18));
		ImGui::SetNextWindowSize(ImVec2(panelWidth, h));

		ImGuiWindowFlags windowFlags = 0;
		windowFlags |= ImGuiWindowFlags_NoTitleBar;
		windowFlags |= ImGuiWindowFlags_NoMove;
		windowFlags |= ImGuiWindowFlags_NoScrollbar;
		windowFlags |= ImGuiWindowFlags_NoResize;

		ImGui::Begin("Model selection", nullptr, windowFlags);

		if (ImGui::CollapsingHeader("Models", ImGuiTreeNodeFlags_DefaultOpen))
		{
			for (auto model : scene.models)
			{
				bool pushedStyle = false;
				if (selectedSelectable.lock() == model) {
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
					pushedStyle = true;
				}
				if (ImGui::Selectable((model->getName() + "##" + model->id).c_str())) {
					selectedSelectable = model;
				}
				if (pushedStyle) {
					ImGui::PopStyleColor();
				}
			}
		}

		if (ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen))
		{
			for (auto light : scene.lights)
			{
				if (ImGui::Selectable(light->name.c_str())) {

					selectedSelectable = light;
				}
			}
		}

		// Unselect the selected object if the user clicks outside the selectable objects
		if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
		{
			selectedSelectable.reset();
		}

		ImGui::End();
	}

	void GUI::drawRightWindow(RenderSettings& renderSettings)
	{
		int w, h;
		window->getSize(w, h);
		int panelWidth = w / 5;
		ImGui::SetNextWindowPos(ImVec2(w - panelWidth, 18));
		ImGui::SetNextWindowSize(ImVec2(panelWidth, h));

		ImGuiWindowFlags windowFlags = 0;
		windowFlags |= ImGuiWindowFlags_NoTitleBar;
		windowFlags |= ImGuiWindowFlags_NoMove;
		windowFlags |= ImGuiWindowFlags_NoScrollbar;
		windowFlags |= ImGuiWindowFlags_NoResize;

		ImGui::Begin("RightWindow", nullptr, windowFlags);

		ImGui::Text("FPS: %f", ImGui::GetIO().Framerate);

		if (selectedSelectable.expired())
			drawRenderingSettings(renderSettings);
		else
			drawSelectableSettings();

		ImGui::End();
	}

	void GUI::drawGizmos(RenderSettings& renderSettings)
	{
		bool shouldDrawGizmos = false;

		float* modelMatrixPtr = nullptr;
		if (auto lockedSelectedSelectable = selectedSelectable.lock())
		{
			if (auto lockedModel = std::dynamic_pointer_cast<Model>(lockedSelectedSelectable))
			{
				shouldDrawGizmos = true;
				modelMatrixPtr = &(lockedModel->trans.transformMatrix.r->m128_f32[0]);
			}
		}

		if (shouldDrawGizmos)
		{
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::SetDrawlist();

			ImGuizmo::SetRect(0, 0, renderSettings.width, renderSettings.height);

			DirectX::XMMATRIX cameraView = renderSettings.camera.getViewMatrix();

			DirectX::XMMATRIX projectionMatrix = renderSettings.camera.getProjectionMatrix(renderSettings.width, renderSettings.height);
			float* viewMatrixPtr = &(cameraView.r->m128_f32[0]);
			float* projectionMatrixPtr = &(projectionMatrix.r->m128_f32[0]);
			ImGuizmo::Manipulate(viewMatrixPtr, projectionMatrixPtr, ImGuizmo::OPERATION::TRANSLATE, ImGuizmo::MODE::LOCAL, modelMatrixPtr);
		}

	}

	void GUI::drawRenderingSettings(RenderSettings& renderSettings)
	{
		ImGui::Text("Rendering Settings");
		ImGui::Checkbox("Use Multisampling", &renderSettings.useMultiSampling);
		ImGui::Checkbox("Use RayTracing", &renderSettings.useRayTracing);

		ImGui::Text("Camera Settings");
		ImGui::SliderFloat("FOV", &renderSettings.camera.fov, 0, 120);
		ImGui::SliderFloat("Near Plane", &renderSettings.camera.nearPlane, 0.1, 50);
		ImGui::SliderFloat("Far Plane", &renderSettings.camera.farPlane, 0.1, 5000);

	}

	void GUI::drawSelectableSettings()
	{
		
		drawSelectedModelSettings();
	}

	void GUI::drawSelectedModelSettings()
	{
		if (auto lockedModel = std::dynamic_pointer_cast<Model>(selectedSelectable.lock()))
		{
			if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
			{

				float matrixTranslation[3], matrixRotation[3], matrixScale[3];
				//DirectX::XMMatrixDecompose(matrixTranslation)
				ImGuizmo::DecomposeMatrixToComponents(&(lockedModel->trans.transformMatrix.r->m128_f32[0]), matrixTranslation, matrixRotation, matrixScale);

				ImGui::InputFloat3("Translation", matrixTranslation);
				ImGui::InputFloat3("Rotation", matrixRotation);
				ImGui::InputFloat3("Scale", matrixScale);

				ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, &(lockedModel->trans.transformMatrix.r->m128_f32[0]));
			}
		}

	}

	void GUI::drawViewport(RenderSettings& renderSettings)
	{
		ImGui::SetNextWindowPos(ImVec2(0, 18));
		ImGui::SetNextWindowSize(ImVec2(renderSettings.width, renderSettings.height));

		ImGuiWindowFlags windowFlags = 0;
		windowFlags |= ImGuiWindowFlags_NoBackground;
		windowFlags |= ImGuiWindowFlags_NoTitleBar;
		windowFlags |= ImGuiWindowFlags_NoMove;
		windowFlags |= ImGuiWindowFlags_NoScrollbar;
		windowFlags |= ImGuiWindowFlags_NoScrollWithMouse;
		windowFlags |= ImGuiWindowFlags_NoCollapse;
		windowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
		windowFlags |= ImGuiWindowFlags_NoResize;

		ImGui::Begin("Viewport", nullptr, windowFlags);

		drawGizmos(renderSettings);

		ImGui::End();
	}

	void GUI::drawTopMenu()
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::BeginMenu("Change scene"))
			{
				for (auto& file : ResourceManager::getAllSceneNames())
				{
					if (ImGui::MenuItem(file.c_str()))
					{
						// Save the current scene
						Serializer::serializeScene(scene);

						Serializer::deserializeScene(file, scene);
					}
				}
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("Create new scene"))
			{
				// Save the current scene
				Serializer::serializeScene(scene);
				Scene newScene{};
				// TODO: Should check for name clashes
				scene = newScene;
				scene.sceneName = "New Scene";
			}
			if (ImGui::MenuItem("Save scene"))
			{
				Serializer::serializeScene(scene);
			}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Models"))
		{
			if (ImGui::MenuItem("Add model from obj file"))
			{
				char fileFilter[64] = "obj files: .obj\0*.obj*\0\0";
				std::string filename = ResourceManager::addFileFromWindowsExplorerToAssets(fileFilter);
				std::shared_ptr<Model> model = std::make_shared<Model>(filename);
				scene.models.push_back(model);
			}
			if (ImGui::BeginMenu("Create Model Primative"))
			{
				if (ImGui::MenuItem("Cube"))
				{
					scene.models.push_back(Model::createPrimative(PrimitiveModelType::CUBE));
				}
				if (ImGui::MenuItem("Sphere"))
				{
					scene.models.push_back(Model::createPrimative(PrimitiveModelType::SPHERE));
				}
				if (ImGui::MenuItem("Plane"))
				{
					scene.models.push_back(Model::createPrimative(PrimitiveModelType::PLANE));
				}
				ImGui::EndMenu();
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Lights"))
		{
			if (ImGui::MenuItem("Add new light"))
			{

			}

			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}
}