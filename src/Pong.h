#pragma once

#include "Scene.h"
#include "RenderSettings.h"
#include "Event.h"
#include "KeyEvent.h"
#include <chrono>

#include "imgui.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

namespace pathtracex {

	class Pong : public IEventListener {
	public:
		Pong()
			: aiModel(nullptr), userModel(nullptr)
			, ballModel(nullptr)
			, blueLight(nullptr), yellowLight(nullptr)
		{}
		~Pong() {}

		void initGame();
		void drawGui();
		void everyFrame();

		virtual void onEvent(Event& e) override;
		bool handleKeyPess(KeyPressedEvent& e);
		bool handleKeyRelease(KeyReleasedEvent& e);
		bool handleMouseMove(MouseMovedEvent& e);
		bool mouseButtonPress(MouseButtonPressedEvent& e);
		bool mouseButtonRelease(MouseButtonReleasedEvent& e);
		
		Scene scene;
		Camera camera{};
		RenderSettings renderSettings{ 0, 0, camera };
	
	private:
		void initScene();
		void updatePlayerMovement();
		void updateAIMovement();
		void updateBallMovement();
		void updateLightMovement();
		void resetRound();
		void startRound();

		void drawPoints();
		void drawPlayerNames();

		ImVec4 textColor{};
		ImFont* textFont;
		float textFontSize = 72.f;

		bool playerUp = false;
		bool playerDown = false;

		bool camUp = false;
		bool camDown = false;
		bool camLeft = false;
		bool camRight = false;
		bool camForward = false;
		bool camBackward = false;

		bool playerMissed = false;
		bool aiMissed = false;
		bool roundHasStarted = false;
		bool gameIsPaused = false;
		float playerSpeed = 10.f;
		float ballSpeed = 8.0f;
		float lightRotDir = 1.0;

		int userPoints = 0;
		int aiPoints = 0;

		float2 ballDir = float2(0, 0);
		std::chrono::time_point<std::chrono::steady_clock> lastUpdate =
			std::chrono::steady_clock::now();
		std::chrono::milliseconds currDelta;
		std::chrono::time_point<std::chrono::steady_clock> roundStart;

		std::shared_ptr<Model> aiModel;
		std::shared_ptr<Model> userModel;
		std::shared_ptr<Model> ballModel;
		std::shared_ptr<Light> yellowLight;
		std::shared_ptr<Light> blueLight;
		//Transform* yellowTrans;
		//Transform* blueTrans;
		DirectX::XMMATRIX defaultCamRotMat;
		DirectX::XMMATRIX defaultCamPosMat;
	};
}