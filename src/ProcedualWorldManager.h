#pragma once
#include "Model.h"
#include "Camera.h"
#include <unordered_map>
#include <memory>

namespace pathtracex
{
	struct ProcedualWorldSettings
	{
		int seed;
		float chunkSideLength = 10;
		float tessellationFactor;
		int chunkRenderDistance = 20;
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
	using CordinateMap = std::unordered_map<Cordinate, std::shared_ptr<Model>, pair_hash>;

	class ProcedualWorldManager
	{
	public:
		ProcedualWorldManager(ProcedualWorldSettings settings) : settings(settings) {};

		void updateProcedualWorld(Camera& camera);

		// The active procedual world models that will be rendered
		std::vector<std::shared_ptr<Model>> procedualWorldModels;
	private: 



		// Mapping of chunk coordinates to model
		// Currently not deleting models, just caching them
		// If memory usage becomes a problem, we can limit the cache size
		CordinateMap procedualWorldModelMap{};


		std::pair<int, int> getChunkCoordinatesAtPosition(const float3 position);

		void createProcedualWorldModel(const std::pair<int, int>& chunkCoordinates);

		void updateProcedualWorldSettings(const ProcedualWorldSettings settings);

		bool settingsChanged = false;

		std::pair<int, int> previousCameraCordinates;
		ProcedualWorldSettings settings;
	};
}