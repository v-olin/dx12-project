#include "Camera.h"

namespace pathtracex {



    DirectX::XMMATRIX Camera::getViewMatrix() const
    {
        return DirectX::XMMatrixInverse(nullptr, transform.transformMatrix);
    }

    DirectX::XMMATRIX Camera::getProjectionMatrix(int width, int height) const
    {
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
        return DirectX::XMMatrixPerspectiveFovLH(DirectX::XMConvertToRadians(fov), aspectRatio, nearPlane, farPlane);
    }

}