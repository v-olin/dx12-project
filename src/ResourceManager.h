#pragma once
#include <string>
#include <vector>

namespace pathtracex {
	class ResourceManager {
	public:
		static std::string addFileFromWindowsExplorerToAssets(char* fileExplorerFilter);

		static std::vector<std::string> getAllSceneNames();
	};
}