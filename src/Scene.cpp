#include "Scene.h"

namespace pathtracex {

	void Scene::initializeModelBuffers() {
		uint32_t modelOffset = 0;
		for (auto& model : models) {
			for (auto& mesh : model->meshes) {
				for (uint32_t i = 0; i < mesh.numberOfVertices; i++) {
					model->vertices[i + mesh.startIndex].materialIdx =
						mesh.materialIdx + modelOffset;
				}
			}

			model->vertexBuffer = std::make_unique<DXVertexBuffer>(model->vertices);
			model->indexBuffer = std::make_unique<DXIndexBuffer>(model->indices);

			modelOffset += model->meshes.size();
		}
	}
}