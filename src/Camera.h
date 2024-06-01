#pragma once

#include "Event.h"
#include "KeyEvent.h"
#include "MouseEvent.h"
#include "Transform.h"
#include <chrono>

namespace pathtracex {
	class Camera : public IEventListener {
	public:
		Transform transform{};

		float fov = 50.0f;
		float nearPlane = 0.1f;
		float farPlane = 100.0f;

		virtual void onEvent(Event& e) override;
		bool handleKeyDown(KeyPressedEvent& e);
		bool handleKeyUp(KeyReleasedEvent& e);
		bool mouseButtonPress(MouseButtonPressedEvent& e);
		bool mouseButtonRelease(MouseButtonReleasedEvent& e);
		bool handleMouseMove(MouseMovedEvent& e);
		void updateMovement();

		DirectX::XMMATRIX getViewMatrix() const;
		DirectX::XMMATRIX getProjectionMatrix(int width, int height) const;
		DirectX::XMMATRIX getJitteredProjectionMatrix(int width, int height) const;
		
	private:
		
		struct Movement {
			bool left{ false };
			bool right{ false };
			bool forward{ false };
			bool backward{ false };
			bool up{ false };
			bool down{ false };
			bool trackingMouse{ false };
		};

		Movement movement;
		std::chrono::steady_clock::time_point lastMovement{ std::chrono::steady_clock::now() };
	};
}