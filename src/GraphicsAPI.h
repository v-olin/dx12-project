#pragma once
#include <DirectXMath.h>
namespace dx = DirectX;

namespace pathtracex {
	enum class GraphicsAPIType {
		DirectX12,
	};

	// Low level wrapper for the underlying graphics API
	class GraphicsAPI {
	public:
		virtual void initGraphicsAPI() = 0;

		virtual void setClearColor(const dx::XMFLOAT3& color) = 0;

		virtual void setCullFace(bool enabled) = 0;

		virtual void setDepthTest(bool enabled) = 0;

		virtual void setDrawTriangleOutline(bool enabled) = 0;

		virtual void setViewport(int x, int y, int width, int height) = 0;

		virtual GraphicsAPIType getGraphicsAPIType() = 0;
	};
}