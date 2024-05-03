#pragma once
#include <vector>
#include <memory>
#include "Model.h"
#include "Camera.h"
#include "RenderSettings.h"
#include "Light.h"
#include "ProcedualWorldManager.h"

namespace pathtracex {
	class Scene {
	public:
		std::vector<std::shared_ptr<Model>> models;
		std::vector<std::shared_ptr<Light>> lights;

		std::vector<std::shared_ptr<Model>> proceduralGroundModels;
		std::vector<std::shared_ptr<Model>> proceduralSkyModels;

		std::string sceneName = "Scene";

		//Camera camera{};
		ProcedualWorldManager* procedualWorldManager;

		/*
		void initializeModelBuffers() {
			uint32_t modelOffset = 0;
			for (auto& model : models) {
				for (auto& mesh : model->meshes) {
					std::vector<uint32_t> meshIdxs{};
					std::vector<Vertex> meshVerts{};

					for (uint32_t i = 0; i < mesh.numberOfVertices; i++) {
						Vertex& vert = model->vertices.at(i + mesh.startIndex);
						vert.materialIdx = mesh.materialIdx + modelOffset;
						meshVerts.push_back(model->vertices.at(i + mesh.startIndex));
						meshIdxs.push_back(i);
					}

					meshVerts.resize(mesh.numberOfVertices);
					meshVerts.insert(
						meshVerts.begin(),
						model->vertices.begin() + mesh.startIndex,
						model->vertices.begin() + mesh.startIndex + mesh.numberOfVertices
					);

					mesh.vbuffer = std::make_shared<DXVertexBuffer>(meshVerts);
					mesh.ibuffer = std::make_shared<DXIndexBuffer>(meshIdxs);
				}

				model->vertexBuffer = std::make_unique<DXVertexBuffer>(model->vertices);
				model->indexBuffer = std::make_unique<DXIndexBuffer>(model->indices);
				modelOffset += model->meshes.size();
			}
		}
		*/
	};
}