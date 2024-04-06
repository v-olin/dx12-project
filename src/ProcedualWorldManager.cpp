#include "ProcedualWorldManager.h"

namespace pathtracex
{
	void ProcedualWorldManager::updateProcedualWorld(Camera& camera)
	{
		// If the camera has not moved to a different chunk, we do not need to update the world
		if (previousCameraCordinates == getChunkCoordinatesAtPosition(camera.transform.getPosition()) && !settingsChanged)
		{
			//return;
		}

		settingsChanged = false;

		procedualWorldModels.clear();

		std::pair<int, int> cord = { -1, 0 };
/*

		if (procedualWorldModelMap.find(cord) != procedualWorldModelMap.end())
			procedualWorldModels.push_back(procedualWorldModelMap[cord]);
		else
			createProcedualWorldModel(cord);

		return;
		*/

		int renderDistanceInChunks = settings.chunkRenderDistance / settings.chunkSideLength;
		for (int x = -renderDistanceInChunks; x < renderDistanceInChunks; x++)
		{
			for (int z = -renderDistanceInChunks; z < renderDistanceInChunks; z++)
			{
				std::pair<int, int> chunkCoordinates = std::pair<int, int>(x, z);
				if (procedualWorldModelMap.find(chunkCoordinates) != procedualWorldModelMap.end())
					procedualWorldModels.push_back(procedualWorldModelMap[chunkCoordinates]);
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
		std::shared_ptr<Model> model = Model::createPrimative(PrimitiveModelType::PLANE);
		model->trans.setScale(float3(settings.chunkSideLength, 1, settings.chunkSideLength));
		model->trans.setPosition(float3((chunkCoordinates.first) * settings.chunkSideLength + settings.chunkSideLength / 2, 0, (chunkCoordinates.second) * settings.chunkSideLength + settings.chunkSideLength / 2));

		procedualWorldModels.push_back(model);
		procedualWorldModelMap[chunkCoordinates] = model;
	}

	void ProcedualWorldManager::updateProcedualWorldSettings(const ProcedualWorldSettings settings)
	{
		settingsChanged = true;
		this->settings = settings;
	}
}