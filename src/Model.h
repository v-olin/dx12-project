// Model.h - Description
#pragma once
#include "Helper.h"
#include "Selectable.h"
#include "Transform.h"
#include "Texture.h"
#include "Material.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "../../vendor/SimpleMath/SimpleMath.h"
#include <DirectXPackedVector.h>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "Transform.h"
#include "Helper.h"
#include "DXIndexBuffer.h"
#include "DXVertexBuffer.h"
#include "Vertex.h"

#include "Vertex.h"

#include "DXVertexBuffer.h"
#include "DXIndexBuffer.h"


namespace pathtracex {

	enum PrimitiveModelType
	{
		CUBE,
		SPHERE,
		CYLINDER,
		PLANE,
		NONE
	};

	struct Mesh : public Selectable {
		std::string name;
		uint32_t materialIdx;
		// Where this Mesh's vertices start
		uint32_t startIndex;
		uint32_t numberOfVertices;
		std::string getName() override { return name; };
	};

	class Model : public Selectable {
	public:
		Model(std::string path);
		//Model(std::shared_ptr<Model> src);
		Model(std::string name
			, std::vector<Material> materials
			, std::vector<Mesh> meshes
			, bool hasDedicatedShader
			, float3 max_cords
			, float3 min_cords
			, std::vector<Vertex> vertices
			, std::vector<uint32_t> indices);
		~Model();
		static std::shared_ptr<Model> createPrimative(PrimitiveModelType type);

		std::string getName() override { return name; };

		Transform trans;



		// The name of the whole model
		std::string name;
		// The filename of this model
		std::string filename;
		// The materials
		std::vector<Material> materials;
		// A model will contain one or more "Meshes"
		std::vector<Mesh> meshes;

		//min and max values in all dimentions, for AABBs
		float3 maxCords;
		float3 minCords;

		// Currently storing vertices, if this becomes a memory issue it can probably be removed
		std::vector<Vertex> vertices{};
		std::vector<uint32_t> indices{};

		// Buffers on GPU
		std::unique_ptr<DXVertexBuffer> vertexBuffer;
		std::unique_ptr<DXIndexBuffer> indexBuffer;
	private:
	
		static std::shared_ptr<Model> createCube();
		static std::shared_ptr<Model> createPlane();
		static std::shared_ptr<Model> createSphere(int stacks, int slices);
	};
}