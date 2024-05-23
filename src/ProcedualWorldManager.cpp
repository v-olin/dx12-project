#include "Logger.h"
#include "ProcedualWorldManager.h"
#include "Texture.h"
#include <DXRenderer.h>

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
		procedualWorldTreeModels.clear();

	// TODO: implement proper sun
	//	if (sun == nullptr || settingsChanged)
	//		createSun();

		settingsChanged = false;

		DXRenderer* renderer = DXRenderer::getInstance();
		renderer->setProcWordValues(settings);

		int renderDistanceInChunks = settings.chunkRenderDistance / settings.chunkSideLength;
		int currentX = (int)camera.transform.getPosition().x / settings.chunkSideLength;

		int currentZ = (int)camera.transform.getPosition().z / settings.chunkSideLength;

		for (int x = -renderDistanceInChunks; x < renderDistanceInChunks; x++)
		{
			for (int z = -renderDistanceInChunks; z < renderDistanceInChunks; z++)
			{
				Cordinate chunkCoordinates = Cordinate(currentX + x, currentZ + z);
				if (procedualWorldModelMap.find(chunkCoordinates) != procedualWorldModelMap.end()) {
					std::shared_ptr<Model> groundModel = procedualWorldModelMap[chunkCoordinates].first;
					procedualWorldGroundModels.push_back(groundModel);
					std::vector<std::shared_ptr<Model>> treeModels = procedualWorldModelMap[chunkCoordinates].second;
					procedualWorldTreeModels.insert(procedualWorldTreeModels.end(), treeModels.begin(), treeModels.end());
				}
				else 
					createProcedualWorldModel(chunkCoordinates);
			}
		}


	}
	
	void ProcedualWorldManager::createMaterial() {

		D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
		ZeroMemory(&heapDesc, sizeof(heapDesc));
		heapDesc.NumDescriptors = NUMTEXTURETYPES;
		heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

		DXRenderer* renderer = DXRenderer::getInstance();
		renderer->createTextureDescriptorHeap(heapDesc, &base_mat.mainDescriptorHeap);

		//base_mat.colorTexture.load("../../assets/", "grass_basecolor.jpg", 4, &base_mat.mainDescriptorHeap, COLTEX);
		base_mat.colorTexture.load("../../assets/textures/", "patchy-meadow1_albedo.png", 5, &base_mat.mainDescriptorHeap, COLTEX);
		base_mat.colorTexture2.load("../../assets/textures/", "angele-kamp-g8IEMx8p_z8-unsplash.jpg", 5, &base_mat.mainDescriptorHeap, COLTEX2);
		base_mat.colorTexture3.load("../../assets/textures/", "rock_pitted_mossy_diff_4k.jpg", 5, &base_mat.mainDescriptorHeap, COLTEX3);


	}

	void ProcedualWorldManager::loadTreeVariations()
	{
		return;
		std::vector<std::string> filenames = { "TREES/Arbaro_1.obj", "TREES/Arbaro_2.obj", "TREES/Arbaro_3.obj", "TREES/weeping_willow.obj" };
		LOG_TRACE("Loading tree variations");
		for (std::string filename : filenames)
		{
			auto model = std::make_shared<Model>(filename);
			model->vertexBuffer = std::make_shared<DXVertexBuffer>(model->vertices);
			model->indexBuffer = std::make_shared<DXIndexBuffer>(model->indices);
			model->vertices.clear();
			model->indices.clear();

			treeVariations.push_back(model);
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
		std::shared_ptr<Model> ground_model = Model::createProcedualWorldMesh(chunkPosition, settings.chunkSideLength, settings.tessellationFactor, settings.heightScale, noiseGenerator);
		ground_model->materials.push_back(base_mat);
		ground_model->trans.setScale(float3(1, 1, 1));
		//ground_model->trans.setPosition(float3((chunkCoordinates.first) * settings.chunkSideLength, 0, (chunkCoordinates.second) * settings.chunkSideLength));
		ground_model->trans.setPosition(float3((chunkCoordinates.first) * settings.chunkSideLength + settings.chunkSideLength / 2, 0, (chunkCoordinates.second) * settings.chunkSideLength + settings.chunkSideLength / 2));



		procedualWorldGroundModels.push_back(ground_model);

		// Create trees
		std::vector<std::shared_ptr<Model>> treeModels = createTrees(chunkCoordinates);
		procedualWorldTreeModels.insert(procedualWorldTreeModels.end(), treeModels.begin(), treeModels.end());
		procedualWorldModelMap[chunkCoordinates] = std::make_pair(ground_model, treeModels);

	}


	std::vector<std::shared_ptr<Model>> ProcedualWorldManager::createTrees(const std::pair<int, int> chunkCoordinates)
	{

		float3 chunkPosition = float3((chunkCoordinates.first) * settings.chunkSideLength, 0, (chunkCoordinates.second) * settings.chunkSideLength);
		return Model::createTreeModels(chunkPosition, settings.chunkSideLength, settings.num_trees, settings.stop_flat_trees, settings.min_tree_dist, settings.heightScale, noiseGenerator, settings.stop_flat, treeVariations, settings.min_tree_scale, settings.max_tree_scale);
	}


	void ProcedualWorldManager::updateProcedualWorldSettings(const ProcedualWorldSettings settings)
	{
		settingsChanged = true;
		this->settings = settings;

		DXRenderer* renderer = DXRenderer::getInstance();
		renderer->setProcWordValues(settings);

		procedualWorldGroundModels.clear();
		procedualWorldModelMap.clear();
	}
}