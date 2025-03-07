#pragma once
#include "Model.h"
#include "Camera.h"
#include "FastNoiseLite.h"
#include <unordered_map>
#include <memory>

namespace pathtracex
{
	struct ProcedualWorldSettings : public Serializable
	{

		float chunkSideLength = 200;
		float tessellationFactor = 200;
		int chunkRenderDistance = 200;
		int heightScale = 8;

		float stop_flat = 0.999;
		float stop_interp = 0.8;
		enum ProcWordColorType
		{
			GRASS,
			SNOW
		};
		int colTexIndex;

		EnumPair colTexIndexEnum = { &colTexIndex,{ "Grass", "Snow" } };

		bool drawProcedualWorldTrees = false;
		int num_trees = 100;
		float stop_flat_trees = 0.95;
		float max_tree_scale = 0.5;
		float min_tree_scale = 0.2;
		float min_tree_dist = 0.001;


		std::vector<SerializableVariable> getSerializableVariables() override
		{
			colTexIndexEnum.val = &colTexIndex;
			return
			{
				{SerializableType::FLOAT, "chunkSideLength", "The side length of a chunk", &chunkSideLength},
				{SerializableType::FLOAT, "tessellationFactor", "The tessellation factor of the procedual world", &tessellationFactor},
				{SerializableType::INT, "chunkRenderDistance", "The render distance of the procedual world", &chunkRenderDistance},
				{SerializableType::INT, "heightScale", "The height scale of the procedual world", &heightScale},
				{SerializableType::FLOAT, "stop_flat", "", &stop_flat},
				{SerializableType::FLOAT, "stop_interp", "", &stop_interp},
				{SerializableType::BOOLEAN, "Draw trees", "", &drawProcedualWorldTrees},
				{SerializableType::INT, "Trees per chunk", "", &num_trees},
				{SerializableType::FLOAT, "Max tree scale", "", &max_tree_scale},
				{SerializableType::FLOAT, "Min tree scale", "", &min_tree_scale},
				{SerializableType::FLOAT, "stop_flat_trees", "", &stop_flat_trees},
				{SerializableType::ENUM, "Flat texture", "", &colTexIndexEnum},
				{SerializableType::FLOAT, "Min tree distance", "", &min_tree_dist}

			};
		};
	};

	// Only for pairs of std::hash-able types for simplicity.
	// You can of course template this struct to allow other hash functions
	struct pair_hash {
		template <class T1, class T2>
		std::size_t operator () (const std::pair<T1, T2>& p) const {
			auto h1 = std::hash<T1>{}(p.first);
			auto h2 = std::hash<T2>{}(p.second);

			// Mainly for demonstration purposes, i.e. works but is overly simple
			// In the real world, use sth. like boost.hash_combine
			return h1 ^ h2;
		}
	};

	using Cordinate = std::pair<int, int>;
	using CordinateMap = std::unordered_map<Cordinate,std::pair<std::shared_ptr<Model>, std::vector<std::shared_ptr<Model>>>, pair_hash>;

	class ProcedualWorldManager
	{
	public:
		ProcedualWorldManager(ProcedualWorldSettings settings) : settings(settings), noiseGenerator() {};

		void updateProcedualWorld(Camera& camera); 
		void createMaterial();
		void loadTreeVariations();

		// The active procedual world models that will be rendered
		std::vector<std::shared_ptr<Model>> procedualWorldGroundModels;

		std::vector<std::shared_ptr<Model>> procedualWorldTreeModels;

		std::vector<std::shared_ptr<Model>> procedualWorldSkyModels;

		//std::shared_ptr<Model> sun = nullptr;

		void updateProcedualWorldSettings(const ProcedualWorldSettings settings);

		ProcedualWorldSettings settings;

		Material base_mat;

		FastNoiseLite noiseGenerator;
	private: 
		// Mapping of chunk coordinates to model
		// Currently not deleting models, just caching them
		// If memory usage becomes a problem, we can limit the cache size
		CordinateMap procedualWorldModelMap{}; 

		bool treeVariationsLoaded = false;
		std::vector<std::shared_ptr<Model>> treeVariations;


		std::pair<int, int> getChunkCoordinatesAtPosition(const float3 position);

		void createProcedualWorldModel(const std::pair<int, int>& chunkCoordinates);
		std::vector<std::shared_ptr<Model>> createTrees(const std::pair<int, int> chunkCoordinates);



		bool settingsChanged = true;

		std::pair<int, int> previousCameraCordinates;
	};
}