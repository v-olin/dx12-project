#include "GUI.h"
#include "imgui.h"

#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"
#include "../../vendor/ImGuizmo/ImGuizmo.h"
#include <DirectXMath.h>
#include <imgui_internal.h>
#include "ResourceManager.h"
#include "Serializer.h"
#include "Logger.h"

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
				bool pushedStyle = false;
				if (selectedSelectable.lock() == light) {
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
					pushedStyle = true;
				}
				if (ImGui::Selectable(light->name.c_str())) {

					selectedSelectable = light;
				}
				if (pushedStyle) {
					ImGui::PopStyleColor();
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

		ImGui::Text(scene.sceneName.c_str());

		ImGui::Text("FPS: %f", ImGui::GetIO().Framerate);

		int drawnModels = DXRenderer::getInstance()->getModelsDrawn();
		ImGui::Text("Drawn models: %d ", drawnModels);
		ImGui::Text("Culled models: %d", scene.models.size() - drawnModels);

		ImGui::Separator();

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
			else if (auto lockedLight = std::dynamic_pointer_cast<Light>(lockedSelectedSelectable))
			{
				shouldDrawGizmos = true;
				modelMatrixPtr = &(lockedLight->transform.transformMatrix.r->m128_f32[0]);
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
		ImGui::Checkbox("Use VSYNC", &renderSettings.useVSYNC);
		ImGui::Checkbox("Use Multisampling", &renderSettings.useMultiSampling);
		ImGui::Checkbox("Draw Bounding Box", &renderSettings.drawBoundingBox);
		ImGui::Checkbox("Use Frustum Culling", &renderSettings.useFrustumCulling);
		ImGui::Checkbox("Use Blooming Effect", &renderSettings.useBloomingEffect);
		ImGui::Checkbox("Use TAA", &renderSettings.useTAA);
		if (renderSettings.raytracingSupported) {
			ImGui::Checkbox("Use RayTracing", &renderSettings.useRayTracing);
		}
		else {
			ImGui::BeginDisabled();
			ImGui::Checkbox("Use RayTracing", &renderSettings.useRayTracing);
			ImGui::EndDisabled();
		}
		ImGui::Checkbox("Draw Procedual World", &renderSettings.drawProcedualWorld);

		ImGui::Separator();

		ImGui::Text("Camera Settings");
		ImGui::SliderFloat("FOV", &renderSettings.camera.fov, 0, 120);
		ImGui::SliderFloat("Near Plane", &renderSettings.camera.nearPlane, 0.1, 50);
		ImGui::SliderFloat("Far Plane", &renderSettings.camera.farPlane, 0.1, 1000.0);
		drawTransformSettings(renderSettings.camera.transform);

		ImGui::Separator();

		ImGui::Text("Procedual world settings");
		static ProcedualWorldSettings procedualWorldSettings = scene.procedualWorldManager->settings;
		drawSerializableVariables(&procedualWorldSettings);

		if (ImGui::CollapsingHeader("Noise settings", ImGuiTreeNodeFlags_None)) {
			drawSerializableVariables(&scene.procedualWorldManager->noiseGenerator);
		}

		if (ImGui::Button("Update Procedual World"))
		{
			scene.procedualWorldManager->updateProcedualWorldSettings(procedualWorldSettings);
		}
		
	}

	void GUI::drawSelectableSettings()
	{
		if (auto lockedModel = std::dynamic_pointer_cast<Model>(selectedSelectable.lock()))
		{
			if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
			{
				drawTransformSettings(lockedModel->trans);
			}

			if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
			{
				for (auto& material : lockedModel->materials)
				{
					if (ImGui::CollapsingHeader(material.name.c_str()))
					{
						drawSerializableVariables(&material);
					}
				}
			}

			ImGui::NewLine();

			drawSerializableVariables(lockedModel.get());
		}
		else if (auto lockedLight = std::dynamic_pointer_cast<Light>(selectedSelectable.lock()))
		{
			if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
			{
				drawTransformSettings(lockedLight->transform);
			}
			drawSerializableVariables(lockedLight.get());
		}

		ImGui::NewLine();

		if (ImGui::Button("Delete Model"))
		{
			if (auto lockedModel = std::dynamic_pointer_cast<Model>(selectedSelectable.lock()))
			{
				scene.models.erase(std::remove(scene.models.begin(), scene.models.end(), lockedModel), scene.models.end());
			}
			else if (auto lockedLight = std::dynamic_pointer_cast<Light>(selectedSelectable.lock()))
			{
				scene.lights.erase(std::remove(scene.lights.begin(), scene.lights.end(), lockedLight), scene.lights.end());
			}
			selectedSelectable.reset();
		}
	}

	void GUI::drawTransformSettings(Transform& transform)
	{
		float matrixTranslation[3], matrixRotation[3], matrixScale[3];
		//DirectX::XMMatrixDecompose(matrixTranslation)
		ImGuizmo::DecomposeMatrixToComponents(&(transform.transformMatrix.r->m128_f32[0]), matrixTranslation, matrixRotation, matrixScale);

		ImGui::InputFloat3("Translation", matrixTranslation);
		ImGui::InputFloat3("Rotation", matrixRotation);
		ImGui::InputFloat3("Scale", matrixScale);

		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, &(transform.transformMatrix.r->m128_f32[0]));
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
				// I am lazy so we will just randomize the name ending to avoid name clashes
				newScene.sceneName = "New Scene " + StringUtil::generateRandomString(5);
				scene = newScene;

				// Serialize the new scene
				Serializer::serializeScene(scene);
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
				scene.lights.push_back(std::make_shared<Light>());
			}

			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}

	void GUI::drawSerializableVariables(Serializable* serializable)
	{
		auto variables = serializable->getSerializableVariables();
		for (auto seralizableVariable : variables)
		{
			if (seralizableVariable.data == nullptr)
			{
				ImGui::Text((seralizableVariable.name + ":").c_str());

				continue;
			}

			if (seralizableVariable.type == SerializableType::STRING)
			{
				std::string data = *static_cast<std::string*>(seralizableVariable.data);
				ImGui::Text((seralizableVariable.name + ":").c_str());
				ImGui::SameLine();
				char name[64];
				memcpy(name, data.c_str(), 64);
				ImGui::InputText(("##" + seralizableVariable.name).c_str(), name, IM_ARRAYSIZE(name));
				*static_cast<std::string*>(seralizableVariable.data) = name;
			}
			else if (seralizableVariable.type == SerializableType::INT)
			{
				ImGui::InputInt(seralizableVariable.name.c_str(), (int*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::FLOAT)
			{
				ImGui::InputFloat(seralizableVariable.name.c_str(), (float*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::BOOLEAN)
			{
				ImGui::Checkbox(seralizableVariable.name.c_str(), (bool*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::DOUBLE)
			{
				ImGui::InputDouble(seralizableVariable.name.c_str(), (double*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::VECTOR2)
			{
				ImGui::InputFloat2(seralizableVariable.name.c_str(), (float*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::VECTOR3)
			{
				ImGui::InputFloat3(seralizableVariable.name.c_str(), (float*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::COLOR)
			{
				ImGui::ColorEdit3(seralizableVariable.name.c_str(), (float*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::VECTOR4)
			{
				ImGui::InputFloat4(seralizableVariable.name.c_str(), (float*)seralizableVariable.data);
			}
			else if (seralizableVariable.type == SerializableType::ENUM)
			{
				std::string name = seralizableVariable.name;
				EnumPair* p = (EnumPair*)seralizableVariable.data;
				if (ImGui::BeginCombo(name.c_str(), p->items[*p->val].c_str())) {
					for (size_t i = 0; i < p->items.size(); ++i) {
						bool isSelected = (*p->val == i);
						if (ImGui::Selectable(p->items[i].c_str(), isSelected)) {
							*p->val = i;
						}
						if (isSelected) {
							ImGui::SetItemDefaultFocus();
						}
					}
					ImGui::EndCombo();
				}

			}
			//else {
			//	return;
			//}

			ImGui::SameLine();
			drawHelpMarker(seralizableVariable.description.c_str());
		}
	}

	void GUI::drawHelpMarker(const char* desc)
	{
		ImGui::TextDisabled("(?)");
		if (ImGui::BeginItemTooltip())
		{
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(desc);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

}