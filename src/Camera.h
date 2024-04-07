#pragma once

#include "Event.h"
#include "KeyEvent.h"
#include "MouseEvent.h"
#include "Transform.h"

namespace pathtracex {
	class Camera : public IEventListener {
	public:
		Transform transform{};

		float fov = 50.0f;
		float nearPlane = 0.1f;
		float farPlane = 1000.0f;

		virtual void onEvent(Event& e) override;
		bool move(KeyPressedEvent& e);
		bool look(MouseMovedEvent& e);
		bool mouseButtonPress(MouseButtonPressedEvent& e);
		bool mouseButtonRelease(MouseButtonReleasedEvent& e);

		DirectX::XMMATRIX getViewMatrix() const;
		DirectX::XMMATRIX getProjectionMatrix(int width, int height) const;
	private:
		bool trackingMouse{ false };
	};
}