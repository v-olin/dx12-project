#include "DXConstantBuffer.h"


namespace pathtracex {
	DXConstantBuffer::DXConstantBuffer(size_t size) : bufferSize(size) {
		allocateBuffer();
	}

	DXConstantBuffer::~DXConstantBuffer() {
		constantBuffer->Release();
	}

	void DXConstantBuffer::allocateBuffer() {

	}
}