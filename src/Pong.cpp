#include "Pong.h"
#include "Model.h"
#include "Serializer.h"
#include <math.h>

#define CLAMP(v,minv,maxv) (std::min(maxv, std::max(minv, v)))

namespace pathtracex {

	void Pong::initGame() {
		initScene();

		renderSettings.useVSYNC = true;
		renderSettings.useRayTracing = true;
		renderSettings.camera.farPlane = 30.f;
		renderSettings.camera.transform.rotate(float3(0, 1, 0), 90.f * 3.1415926535f / 180.0f);
		renderSettings.camera.transform.setPosition(float3(0, 0.12f, -13.f));
		renderSettings.isPongGame = true;

		ImGuiIO& io = ImGui::GetIO();
		textColor = ImVec4(0.43f, 0.2f, 0.92f, 1.0f);
		textFont = io.Fonts->AddFontFromFileTTF("../../assets/fonts/roboto.ttf", textFontSize);
	}

	void Pong::drawGui() {
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		drawPoints();
		drawPlayerNames();

		ImGui::Render();
	}

	void Pong::drawPoints() {
		int w = renderSettings.width;
		int h = renderSettings.height;

		int pw = 125;
		int ph = 100;

		ImGui::SetNextWindowPos(ImVec2((w - pw) / 2, 30));
		ImGui::SetNextWindowSize(ImVec2(pw, ph));

		ImGuiWindowFlags wflags = 0;
		wflags |= ImGuiWindowFlags_NoTitleBar;
		wflags |= ImGuiWindowFlags_NoMove;
		wflags |= ImGuiWindowFlags_NoScrollbar;
		wflags |= ImGuiWindowFlags_NoResize;
		wflags |= ImGuiWindowFlags_NoBackground;

		ImGui::Begin("Points", nullptr, wflags);

		const ImVec4 tcolor = textColor;
		ImGui::PushStyleColor(ImGuiCol_Text, tcolor);
		ImGui::PushFont(textFont);
		ImGui::Text("%d:%d", aiPoints, userPoints);
		ImGui::PopStyleColor();
		ImGui::PopFont();

		ImGui::End();
	}

	void Pong::drawPlayerNames() {

	}

	void Pong::initScene() {
		Serializer::deserializeScene("pong", scene);

		for (auto& model : scene.models) {
			if (model->name.find(std::string{ "AI" }) != std::string::npos) {
				this->aiModel = model;
			}
			else if (model->name.find(std::string{ "User" }) != std::string::npos) {
				this->userModel = model;
			}
			else if (model->name.find(std::string{ "Ball" }) != std::string::npos) {
				this->ballModel = model;
			}
		}

		for (auto& light : scene.lights) {
			if (light->name.find(std::string{ "Blue" }) != std::string::npos) {
				//this->blueLight = light;
			}
			else if (light->name.find(std::string{ "Yellow" }) != std::string::npos) {
				this->yellowLight = light;
			}
		}
	}

	void Pong::everyFrame() {
		std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
		currDelta = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate);
		lastUpdate = now;

		if (!gameIsPaused && roundHasStarted) {
			updatePlayerMovement();
			updateAIMovement();
			updateBallMovement();
			updateLightMovement();
		}

		drawGui();
		
	}

	void Pong::updatePlayerMovement() {

		if (playerUp) {
			float secDelta = float(currDelta.count()) / 1000.f;
			Transform& transform = userModel->trans;
			transform.translate(float3(0, secDelta * playerSpeed, 0));

			// ensure it is inside box
			float3 newPos = transform.getPosition();
			if (newPos.y > 4.0f) {
				transform.setPosition(float3(newPos.x, 4.0f, newPos.z));
				playerUp = false;
			}
		}

		if (playerDown) {
			float secDelta = float(currDelta.count()) / 1000.f;
			Transform& transform = userModel->trans;
			transform.translate(float3(0, -secDelta * playerSpeed, 0));

			// ensure it is inside box
			float3 newPos = transform.getPosition();
			if (newPos.y < -4.0f) {
				transform.setPosition(float3(newPos.x, -4.0f, newPos.z));
				playerDown = false;
			}
		}
	}

	void Pong::updateAIMovement() {
		float aiHeight = aiModel->trans.getPosition().y;
		float ballHeight = ballModel->trans.getPosition().y;

		float diff = aiHeight - ballHeight;

		if (abs(diff) < FLT_EPSILON * 5 || diff == 0.0f) {
			return;
		}

		// if AI above ball, move it down
		if (diff > 0) {
			float secDelta = float(currDelta.count()) / 1000.f;
			Transform& transform = aiModel->trans;
			transform.translate(float3(0, -secDelta * playerSpeed, 0));

			// ensure it is inside box
			float3 newPos = transform.getPosition();
			if (newPos.y > 4.0f) {
				transform.setPosition(float3(newPos.x, 4.0f, newPos.z));

			}
		}
		// if AI below ball, move it up
		else {
			float secDelta = float(currDelta.count()) / 1000.f;
			Transform& transform = aiModel->trans;
			transform.translate(float3(0, secDelta * playerSpeed, 0));

			// ensure it is inside box
			float3 newPos = transform.getPosition();
			if (newPos.y < -4.0f) {
				transform.setPosition(float3(newPos.x, -4.0f, newPos.z));
			}
		}
	}

	void Pong::updateBallMovement() {
		float d = float(currDelta.count()) / 100.f;
		Transform& transform = ballModel->trans;
		transform.translate(float3(ballDir.x, ballDir.y, 0.f) * d * ballSpeed);

		float3 newPos = ballModel->trans.getPosition();

		// ball collided with bottom plane
		if (newPos.y < -4.5f) {
			transform.setPosition(float3(newPos.x, -4.5f, newPos.z));
			ballDir = float2(ballDir.x, ballDir.y * -1.0f);
			lightRotDir *= -1.0f;
		}
		// ball collided with top plane
		else if (newPos.y > 4.5f) {
			transform.setPosition(float3(newPos.x, 4.5f, newPos.z));
			lightRotDir *= -1.0f;
			ballDir = float2(ballDir.x, ballDir.y * -1.0f);
		}

		const float ballRad = 0.5f;
		const float halfBoxWidth = 0.25f;
		const float halfBoxHeight = 1.0f;
		float ballHeight = newPos.y;
		// check collision with player
		if (newPos.x > (9.0f - ballRad - halfBoxWidth) && !playerMissed) {
			float playerHeight = userModel->trans.getPosition().y;

			// player hits ball
			if (ballHeight >= playerHeight - halfBoxHeight && ballHeight <= playerHeight + halfBoxHeight) {
				transform.setPosition(float3(8.25f, newPos.y, newPos.z));
				lightRotDir *= -1.0f;
				ballDir = float2(ballDir.x * -1.f, ballDir.y);
			}
			else {
				playerMissed = true;
			}
		}
		else if (newPos.x < (-9.0f + ballRad + halfBoxWidth) && !aiMissed) {
			float aiHeight = aiModel->trans.getPosition().y;

			if (ballHeight >= aiHeight - halfBoxHeight && ballHeight <= aiHeight + halfBoxHeight) {
				transform.setPosition(float3(-8.25f, newPos.y, newPos.z));
				lightRotDir *= -1.0f;
				ballDir = float2(ballDir.x * -1.0f, ballDir.y);
			}
			else {
				aiMissed = true;
			}
		}

		if (newPos.x > 10.0f && playerMissed) {
			resetRound();
			playerMissed = false;
			++aiPoints;
		}
		else if (newPos.x < -10.0f && aiMissed) {
			resetRound();
			aiMissed = false;
			++userPoints;
		}
	}

	void Pong::updateLightMovement() {
		std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
		std::chrono::milliseconds lightDelta = std::chrono::duration_cast<std::chrono::milliseconds>(now - roundStart);

		float d = float(lightDelta.count()) / 700.f;
		
		// rotate yellow/right light
		{
			Transform& trans = yellowLight->transform;
			float cosp = cosf(d) * -2.0f;
			float sinp = sinf(d) * -2.0f;

			float3 pos = ballModel->trans.getPosition();
			float xp = std::min(-4.7f, std::max(4.7f, pos.x + cosp));
			trans.setPosition(float3(CLAMP(pos.x + cosp, -8.5f, 8.5f), CLAMP(pos.y + sinp, -4.7f, 4.7f), pos.z));
		}
	}

	void Pong::resetRound() {
		aiModel->trans.setPosition(float3(-9, 0, 0));
		userModel->trans.setPosition(float3(9, 0, 0));
		ballModel->trans.setPosition(float3(0, 0, 0));

		static const float r = 1.0f / sqrt(2.0f);
		static const float2 possibleDirs[4]{
			float2(r,r),
			float2(-r,r),
			float2(-r,-r),
			float2(r,-r)
		};

		ballDir = possibleDirs[rand() % 4];
		roundHasStarted = false;
		gameIsPaused = false;
	}

	void Pong::startRound() {
		if (roundHasStarted) {
			gameIsPaused = !gameIsPaused;
		}
		else {
			roundHasStarted = true;
		}
	}

	void Pong::onEvent(Event& e) {
		EventDispatcher dispatcher{ e };

		if (e.getEventType() == EventType::KeyPressed) {
			dispatcher.dispatch<KeyPressedEvent>(BIND_EVENT_FN(Pong::handleKeyPess));
		}
		else if (e.getEventType() == EventType::KeyReleased) {
			dispatcher.dispatch<KeyReleasedEvent>(BIND_EVENT_FN(Pong::handleKeyRelease));
		}
		else if (e.getEventType() == EventType::MouseMoved) {
			dispatcher.dispatch<MouseMovedEvent>(BIND_EVENT_FN(Pong::handleMouseMove));
		}
	}

	bool Pong::handleKeyPess(KeyPressedEvent& e) {
		auto key = e.getKeyCode();
		
		switch (key) {
		case 'W':
		case 'w':
			playerUp = true;
			return true;
		case 'S':
		case 's':
			playerDown = true;
			return true;
		case 'R':
		case 'r':
			resetRound();
			return true;
		case 'P':
		case 'p':
			startRound();
			return true;
		case 'M':
		case 'm':
			renderSettings.useRayTracing = !renderSettings.useRayTracing;
			return true;
		default:
			break;
		}

		return false;
	}

	bool Pong::handleKeyRelease(KeyReleasedEvent& e) {
		switch (e.getKeyCode()) {
		case 'W':
		case 'w':
			playerUp = false;
			return true;
		case 'S':
		case 's':
			playerDown = false;
			return true;
		default:
			break;
		}

		return false;
	}

	bool Pong::handleMouseMove(MouseMovedEvent& e) {
		return false;
	}

	bool Pong::mouseButtonPress(MouseButtonPressedEvent& e) {
		return false;
	}

	bool Pong::mouseButtonRelease(MouseButtonReleasedEvent& e) {
		return false;
	}

}