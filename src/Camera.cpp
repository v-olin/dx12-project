#include "Camera.h"

#include "KeyEvent.h"
#include "Logger.h"
#include <sstream>

namespace pathtracex {

    void Camera::onEvent(Event& e) {
        EventDispatcher dispatcher{ e };

        if (e.getEventType() == EventType::KeyPressed) {
            dispatcher.dispatch<KeyPressedEvent>(BIND_EVENT_FN(Camera::handleKeyDown));
        }
        else if (e.getEventType() == EventType::KeyReleased) {
            dispatcher.dispatch<KeyReleasedEvent>(BIND_EVENT_FN(Camera::handleKeyUp));
        }
        else if (e.getEventType() == EventType::MouseMoved) {
            dispatcher.dispatch<MouseMovedEvent>(BIND_EVENT_FN(Camera::handleMouseMove));
        }
        else if (e.getEventType() == EventType::MousePressed) {
            dispatcher.dispatch<MouseButtonPressedEvent>(BIND_EVENT_FN(Camera::mouseButtonPress));
        }
        else if (e.getEventType() == EventType::MouseReleased) {
            dispatcher.dispatch<MouseButtonReleasedEvent>(BIND_EVENT_FN(Camera::mouseButtonRelease));
        }
    }

    bool Camera::handleKeyDown(KeyPressedEvent& e) {
        switch (e.getKeyCode()) {
        case 'W':
        case 'w':
            movement.forward = true;
            break;
        case 'A':
        case 'a':
            movement.left = true;
            break;
        case 'S':
        case 's': 
            movement.backward = true;
            break;
        case 'D':
        case 'd': 
            movement.right = true;
            break;
        case 'Q':
        case 'q':
            movement.up = true;
            break;
        case 'E':
        case 'e': 
            movement.down = true;
            break;
        default:
            break;
        }

        return true; // event handled
    }

    bool Camera::handleKeyUp(KeyReleasedEvent& e) {
        switch (e.getKeyCode()) {
        case 'W':
        case 'w':
            //transform.translate(-1.f * transform.getForward() * sensitivity);
            movement.forward = false;
            break;
        case 'A':
        case 'a':
            //transform.translate(-1.f * transform.getRight() * sensitivity);
            movement.left = false;
            break;
        case 'S':
        case 's':
            //transform.translate(transform.getForward() * sensitivity);
            movement.backward = false;
            break;
        case 'D':
        case 'd':
            movement.right = false;
            //transform.translate(transform.getRight() * sensitivity);
            break;
        case 'Q':
        case 'q':
            movement.up = false;
            //transform.translate(transform.getUp() * sensitivity);
            break;
        case 'E':
        case 'e':
            movement.down = false;
            //transform.translate(-1.f * transform.getUp() * sensitivity);
            break;
        default:
            break;
        }

        return true;
    }

    bool Camera::handleMouseMove(MouseMovedEvent& e) {
        if (!movement.trackingMouse) {
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
            movement.trackingMouse = true;
        }

        return true;
    }

    bool Camera::mouseButtonRelease(MouseButtonReleasedEvent& e) {
        if (e.getMouseButton() == MouseButtonType::LeftButton) {
            movement.trackingMouse = false;
        }

        return true;
    }

    void Camera::updateMovement() {
        static const float sensitivity = 0.05f;

        std::chrono::duration<float> diff = std::chrono::steady_clock::now() - lastMovement;
        auto diffms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();

        if (diffms < 17) {
            return; // update camera ~60 fps
        }
        
        if (movement.forward) {
            transform.translate(-1.f * transform.getForward() * sensitivity);
        }
        if (movement.left) {
            transform.translate(-1.f * transform.getRight() * sensitivity);
        }
        if (movement.backward) {
            transform.translate(transform.getForward() * sensitivity);
        }
        if (movement.right) {
            transform.translate(transform.getRight() * sensitivity);
        }
        if (movement.up) {
            transform.translate(transform.getUp() * sensitivity);
        }
        if (movement.down) {
            transform.translate(-1.f * transform.getUp() * sensitivity);
        }

        lastMovement = std::chrono::steady_clock::now();
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