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

			// Calculate the bounding box verticies for the model
			float3 maxCords = model->getMaxCords();
			float3 minCords = model->getMinCords();

			std::vector<Vertex> boundingBoxVertices = {
							{minCords.x, minCords.y, minCords.z, 255, 0, 0, 1}, // Front bottom left
							{maxCords.x, minCords.y, minCords.z, 255, 0, 0, 1}, // Front bottom right
							{maxCords.x, minCords.y, minCords.z, 255, 0, 0, 1}, // Front bottom right
							{maxCords.x, maxCords.y, minCords.z, 255, 0, 0, 1}, // Front top right
							{maxCords.x, maxCords.y, minCords.z, 255, 0, 0, 1}, // Front top right
							{minCords.x, maxCords.y, minCords.z, 255, 0, 0, 1}, // Front top left
							{minCords.x, maxCords.y, minCords.z, 255, 0, 0, 1}, // Front top left
							{minCords.x, minCords.y, minCords.z, 255, 0, 0, 1}, // Front bottom left

							// Repeat for the back face
							{minCords.x, minCords.y, maxCords.z, 255, 0, 0, 1}, // Back bottom left
							{maxCords.x, minCords.y, maxCords.z, 255, 0, 0, 1}, // Back bottom right
							{maxCords.x, minCords.y, maxCords.z, 255, 0, 0, 1}, // Back bottom right
							{maxCords.x, maxCords.y, maxCords.z, 255, 0, 0, 1}, // Back top right
							{maxCords.x, maxCords.y, maxCords.z, 255, 0, 0, 1}, // Back top right
							{minCords.x, maxCords.y, maxCords.z, 255, 0, 0, 1}, // Back top left
							{minCords.x, maxCords.y, maxCords.z, 255, 0, 0, 1}, // Back top left
							{minCords.x, minCords.y, maxCords.z, 255, 0, 0, 1}, // Back bottom left

							// Connect the front and back faces
							{minCords.x, minCords.y, minCords.z, 255, 0, 0, 1}, // Front bottom left
							{minCords.x, minCords.y, maxCords.z, 255, 0, 0, 1}, // Back bottom left
							{maxCords.x, minCords.y, minCords.z, 255, 0, 0, 1}, // Front bottom right
							{maxCords.x, minCords.y, maxCords.z, 255, 0, 0, 1}, // Back bottom right
							{maxCords.x, maxCords.y, minCords.z, 255, 0, 0, 1}, // Front top right
							{maxCords.x, maxCords.y, maxCords.z, 255, 0, 0, 1}, // Back top right
							{minCords.x, maxCords.y, minCords.z, 255, 0, 0, 1}, // Front top left
							{minCords.x, maxCords.y, maxCords.z, 255, 0, 0, 1}, // Back top left
			};
			model->vertexBufferBoundingBox = std::make_unique<DXVertexBuffer>(boundingBoxVertices);

			modelOffset += model->meshes.size();
		}
	}
}