#include "GUI.h"
#include "imgui.h"

#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"
#include "../../vendor/ImGuizmo/ImGuizmo.h"
#include <DirectXMath.h>


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

	void GUI::drawGUI(RenderSettings& renderSettings)
	{
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		drawTopMenu();
		drawModelSelectionMenu();
		drawRightWindow(renderSettings);

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

		ImGui::Begin("Model selection", nullptr, windowFlags);

		if (ImGui::CollapsingHeader("Models", ImGuiTreeNodeFlags_DefaultOpen))
		{
			for (auto model : scene.models)
			{
				if (ImGui::Selectable(model->getName().c_str())) {

					selectedSelectable = model;
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

		ImGui::Begin("RightWindow", nullptr, windowFlags);

		if (selectedSelectable.expired())
			drawRenderingSettings(renderSettings);
		else
			drawSelectableSettings();

		ImGui::End();
	}

	void GUI::drawGizmos()
	{

	}

	void GUI::drawRenderingSettings(RenderSettings& renderSettings)
	{
		ImGui::Text("Rendering Settings");
		ImGui::Checkbox("Use Multisampling", &renderSettings.useMultiSampling);
		ImGui::Checkbox("Use RayTracing", &renderSettings.useRayTracing);

		ImGui::Text("Camera Settings");
		ImGui::SliderFloat("FOV", &scene.camera.fov, 0, 120);
		ImGui::SliderFloat("Near Plane", &scene.camera.nearPlane, 0, 500);
		ImGui::SliderFloat("Far Plane", &scene.camera.farPlane, 0, 5000);

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
				//ImGuizmo::DecomposeMatrixToComponents(&(lockedModel->transform.transformMatrix.r->m128_f32[0]), matrixTranslation, matrixRotation, matrixScale);

				ImGui::InputFloat3("Translation", matrixTranslation);
				ImGui::InputFloat3("Rotation", matrixRotation);
				ImGui::InputFloat3("Scale", matrixScale);

				//ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, &(lockedModel->transform.transformMatrix.r->m128_f32[0]));
			}
		}

	}

	void GUI::drawTopMenu()
	{
		ImGui::BeginMainMenuBar();

		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Change scene"))
			{

			}
			if (ImGui::MenuItem("Create new scene"))
			{

			}
			if (ImGui::MenuItem("Save scene"))
			{

			}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Models"))
		{
			if (ImGui::MenuItem("Add model from obj file"))
			{

			}
			if (ImGui::MenuItem("Add model primative"))
			{

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