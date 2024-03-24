#pragma once
#include <vector>
#include "DXRenderer.h"

namespace pathtracex {
	class DXVertexBuffer {
		DXVertexBuffer(std::vector<Vertex> vertices);
		~DXVertexBuffer();
		void bind();
	};
}