#include "Camera.h"

#include "KeyEvent.h"
#include "Logger.h"
#include <sstream>

namespace pathtracex {

    void Camera::onEvent(Event& e) {
        EventDispatcher dispatcher{ e };

        if (e.getEventType() == EventType::KeyPressed) {
            dispatcher.dispatch<KeyPressedEvent>(BIND_EVENT_FN(Camera::move));
        }
        else if (e.getEventType() == EventType::MouseMoved) {
            dispatcher.dispatch<MouseMovedEvent>(BIND_EVENT_FN(Camera::look));
        }
        else if (e.getEventType() == EventType::MousePressed) {
            dispatcher.dispatch<MouseButtonPressedEvent>(BIND_EVENT_FN(Camera::mouseButtonPress));
        }
        else if (e.getEventType() == EventType::MouseReleased) {
            dispatcher.dispatch<MouseButtonReleasedEvent>(BIND_EVENT_FN(Camera::mouseButtonRelease));
        }
    }

    bool Camera::move(KeyPressedEvent& e) {
        static const float sensitivity = 0.3f;

        switch (e.getKeyCode()) {
        case 'W':
        case 'w':
            transform.translate(-1.f * transform.getForward() * sensitivity);
            break;
        case 'A':
        case 'a':
            transform.translate(-1.f * transform.getRight() * sensitivity);
            break;
        case 'S':
        case 's': 
            transform.translate(transform.getForward() * sensitivity);
            break;
        case 'D':
        case 'd': 
            transform.translate(transform.getRight() * sensitivity);
            break;
        case 'Q':
        case 'q':
            transform.translate(transform.getUp() * sensitivity);
            break;
        case 'E':
        case 'e': 
            transform.translate(-1.f * transform.getUp() * sensitivity);
            break;
        default:
            break;
        }

        return true; // event handled
    }

    bool Camera::look(MouseMovedEvent& e) {
        if (!trackingMouse) {
            return false;
        }

        static const float sensitivity = 1.f;
        static const float3 worldUp{ 0.f, 1.f, 0.f };

        auto ndx = (float)(e.getDiffX()) / 1000.f;
        auto ndy = (float)(e.getDiffY()) / 1000.f;

        auto yaw = DirectX::XMMatrixRotationAxis(worldUp, ndx * sensitivity);
        transform.rotate(worldUp, ndx * sensitivity);

        auto paxis = DirectX::XMVector3Normalize(DirectX::XMVector3Cross(transform.getForward(), worldUp));
        transform.rotate(paxis, ndy * sensitivity);

        return true;
    }

    bool Camera::mouseButtonPress(MouseButtonPressedEvent& e) {
        if (e.getMouseButton() == MouseButtonType::LeftButton) {
            trackingMouse = true;
        }

        return true;
    }

    bool Camera::mouseButtonRelease(MouseButtonReleasedEvent& e) {
        if (e.getMouseButton() == MouseButtonType::LeftButton) {
            trackingMouse = false;
        }

        return true;
    }

    DirectX::XMMATRIX Camera::getViewMatrix() const
    {
        return DirectX::XMMatrixLookAtRH(
            transform.getPosition(),
            transform.getPosition() + transform.getForward(),
            transform.getUp());

        //return DirectX::XMMatrixInverse(nullptr, DirectX::XMMatrixMultiply(transform.transformMatrix, transform.rotationMatrix));
    }

    DirectX::XMMATRIX Camera::getProjectionMatrix(int width, int height) const
    {
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
        return DirectX::XMMatrixPerspectiveFovLH(DirectX::XMConvertToRadians(fov), aspectRatio, nearPlane, farPlane);
    }

}