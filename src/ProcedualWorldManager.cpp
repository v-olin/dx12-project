#include "ProcedualWorldManager.h"

namespace pathtracex
{
	void ProcedualWorldManager::updateProcedualWorld(Camera& camera)
	{
		// If the camera has not moved to a different chunk, we do not need to update the world
		if (previousCameraCordinates == getChunkCoordinatesAtPosition(camera.transform.getPosition()) && !settingsChanged)
		{
			return;
		}
		procedualWorldGroundModels.clear();

	// TODO: implement proper sun
	//	if (sun == nullptr || settingsChanged)
	//		createSun();

		settingsChanged = false;

		int renderDistanceInChunks = settings.chunkRenderDistance / settings.chunkSideLength;
		int currentX = (int)camera.transform.getPosition().x / settings.chunkSideLength;

		int currentZ = (int)camera.transform.getPosition().z / settings.chunkSideLength;

		for (int x = -renderDistanceInChunks; x < renderDistanceInChunks; x++)
		{
			for (int z = -renderDistanceInChunks; z < renderDistanceInChunks; z++)
			{
				Cordinate chunkCoordinates = Cordinate(currentX + x, currentZ + z);
				if (procedualWorldModelMap.find(chunkCoordinates) != procedualWorldModelMap.end())
					procedualWorldGroundModels.push_back(procedualWorldModelMap[chunkCoordinates]);
				else 
					createProcedualWorldModel(chunkCoordinates);
			}
		}
	}

	std::pair<int, int> ProcedualWorldManager::getChunkCoordinatesAtPosition(const float3 position)
	{
		int x = (int)position.x / settings.chunkSideLength;
		int z = (int)position.z / settings.chunkSideLength;

		return std::pair<int, int>(x, z);
	}
	
	// The 0,0 chunk is placed from 0,0 to chunkSideLength, chunkSideLength
	void ProcedualWorldManager::createProcedualWorldModel(const std::pair<int, int>& chunkCoordinates)
	{	
		float3 chunkPosition = float3((chunkCoordinates.first) * settings.chunkSideLength, 0, (chunkCoordinates.second) * settings.chunkSideLength);
		std::shared_ptr<Model> model = Model::createProcedualWorldMesh(chunkPosition, settings.chunkSideLength, settings.seed, settings.tessellationFactor, settings.heightScale);
		model->trans.setScale(float3(1, 1, 1));
		model->trans.setPosition(float3((chunkCoordinates.first) * settings.chunkSideLength + settings.chunkSideLength / 2, 0, (chunkCoordinates.second) * settings.chunkSideLength + settings.chunkSideLength / 2));

		procedualWorldGroundModels.push_back(model);
		procedualWorldModelMap[chunkCoordinates] = model;
	}

	void ProcedualWorldManager::createSun()
	{
		std::shared_ptr<Model> model = Model::createPrimative(PrimitiveModelType::SPHERE);
		model->trans.setScale(float3(10, 10, 10));
		model->trans.setPosition(float3(0, 500, 500));

		sun = model;
		
		procedualWorldSkyModels.push_back(model);
	}

	void ProcedualWorldManager::updateProcedualWorldSettings(const ProcedualWorldSettings settings)
	{
		settingsChanged = true;
		this->settings = settings;

		procedualWorldGroundModels.clear();
		procedualWorldModelMap.clear();
	}
}