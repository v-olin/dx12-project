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
		Pong() {}
		~Pong() {}

		void initGame();
		void drawGui();
		void everyFrame();

		virtual void onEvent(Event& e) override;
		bool handleKeyPess(KeyPressedEvent& e);
		bool handleKeyRelease(KeyReleasedEvent& e);
		
		Scene scene;
		Camera camera{};
		RenderSettings renderSettings{ 0, 0, camera };
	
	private:
		void initScene();
		void updatePlayerMovement();
		void updateAIMovement();
		void updateBallMovement();
		void resetRound();
		void startRound();

		void drawPoints();
		void drawPlayerNames();

		ImVec4 textColor{};
		ImFont* textFont;
		float textFontSize = 72.f;

		bool playerUp = false;
		bool playerDown = false;
		bool playerMissed = false;
		bool aiMissed = false;
		bool roundHasStarted = false;
		bool gameIsPaused = false;
		float playerSpeed = 10.f;
		float ballSpeed = 8.0f;

		int userPoints = 0;
		int aiPoints = 0;

		float2 ballDir = float2(0, 0);
		std::chrono::time_point<std::chrono::steady_clock> lastUpdate =
			std::chrono::steady_clock::now();
		std::chrono::milliseconds currDelta;

		std::shared_ptr<Model> aiModel;
		std::shared_ptr<Model> userModel;
		std::shared_ptr<Model> ballModel;
	};

}