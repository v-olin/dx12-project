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
#include "StringUtil.h"

#include "DXVertexBuffer.h"
#include "DXIndexBuffer.h"
#include "FastNoiseLite.h"


namespace pathtracex {
	namespace file_util {

		std::string normalise(const std::string& file_name);
		std::string parent_path(const std::string& file_name);
		std::string file_stem(const std::string& file_name);
		std::string file_extension(const std::string& file_name);
		std::string change_extension(const std::string& file_name, const std::string& ext);
	}


	enum PrimitiveModelType
	{
		CUBE,
		SPHERE,
		CYLINDER,
		PLANE,
		CONE,
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
		Model(std::string filenameWithExtension);
		//Model(std::shared_ptr<Model> src);
		Model(std::string name
			, std::vector<Material> materials
			, std::vector<Mesh> meshes
			, bool hasDedicatedShader
			, float3 max_cords
			, float3 min_cords
			, std::vector<Vertex> vertices
			, std::vector<uint32_t> indices);
		Model(std::string name
			, std::vector<Material> materials
			, std::vector<Mesh> meshes
			, bool hasDedicatedShader
			, float3 max_cords
			, float3 min_cords
			, std::shared_ptr<DXVertexBuffer> vertexBuffer
			, std::shared_ptr<DXIndexBuffer> indexBuffer);

		~Model();

		static std::shared_ptr<Model> createPrimative(PrimitiveModelType type);

		static float3 cross(float3 a, float3 b);

		std::string getName() override { return name; };

		static std::string primitiveModelTypeToString(PrimitiveModelType type);
		static PrimitiveModelType stringToPrimitiveModelType(std::string type);

		Transform trans;

		PrimitiveModelType primativeType = PrimitiveModelType::NONE;

		// The name of the whole model
		std::string name;
		// The filename of this model
		std::string filename = "";
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
		std::shared_ptr<DXVertexBuffer> vertexBuffer;
		std::shared_ptr<DXIndexBuffer> indexBuffer;

		std::vector<SerializableVariable> getSerializableVariables() override
		{
			return 
			{
				{SerializableType::STRING, "Name", "The name of the model", &name},
				{SerializableType::STRING, "Filename", "The filename of the model obj file", &filename},
				{SerializableType::MATRIX4X4, "TransformMatrix", "The transform matrix of the model", &trans.transformMatrix}
			};
		};


		//static std::shared_ptr<Model> createProcedualWorldMesh(float3 startPos, float sideLength, int seed, int tesselation, int heightScale = 10, int octaves = 6);
		static std::shared_ptr<Model> createProcedualWorldMesh(float3 startPos, float sideLength, int tesselation, int heightScale, FastNoiseLite nGen);

		//static std::vector<std::shared_ptr<Model>> createTreeModels(float3 startPos, float sideLength, int numTrees, int heightScale, FastNoiseLite nGen, float stop_interp);
		static std::vector<std::shared_ptr<Model>> createTreeModels(float3 startPos, float sideLength, int numTrees, int heightScale, FastNoiseLite nGen, float stop_flat, std::vector<std::shared_ptr<Model>> treeVariations);


		std::string id = StringUtil::generateRandomString(10);
	private:

	
		static std::shared_ptr<Model> createCube();
		static std::shared_ptr<Model> createPlane();
		static std::shared_ptr<Model> createSphere(int stacks, int slices);
		static std::shared_ptr<Model> createCylinder(int baseRadius, int topRadius, int height, int sectorCount);
	};

}