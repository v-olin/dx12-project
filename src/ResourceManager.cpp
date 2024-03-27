#include "ResourceManager.h"
#include "Logger.h"
#include "Windows.h"
#include <iostream>
#include <ShlObj.h>
#include <commdlg.h>
#include <filesystem>

#include <fstream>
#include <sstream>
#include <regex>


namespace fs = std::filesystem;

namespace pathtracex {
#define WIN32_API_ERROR_CODE_FILE_ALREADY_EXISTS 80
#define PATH_TO_ASSETS_FOLDER "../../assets/"

	// copies file from file explorer to subdirectory and returns the file name
	// if no file is selected, returns empty string
	std::string ResourceManager::addFileFromWindowsExplorerToAssets(char* fileExplorerFilter)
	{
		OPENFILENAME ofn = { 0 };
		TCHAR szFile[260] = { 0 };
		ofn.lStructSize = sizeof(ofn);
		ofn.lpstrFile = szFile;
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = fileExplorerFilter;
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = fs::current_path().string().c_str();
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

		if (GetOpenFileName(&ofn) == FALSE)
		{
			LOG_INFO("User Cancelled File Explorer Dialog");
			return "";
		}

		// use ofn.lpstrFile here
		std::cout << ofn.lpstrFile << std::endl;
		fs::path objFilePath = ofn.lpstrFile;
		std::string objFileName = objFilePath.filename().string();
		std::string objFileNameNoExtension = objFilePath.stem().string();
		std::string objFileExtension = objFilePath.extension().string();
		int objFileSize = fs::file_size(objFilePath);

	//	ResourceType type = getResourceTypeFromFileName(objFileName);
		//std::string destinationFolderName = getResourceFolderName(type);

		std::string destinationPath = PATH_TO_ASSETS_FOLDER + objFileName;
		LOG_INFO("Copying file from file explorer to assets: " + destinationPath);
		CopyFile(ofn.lpstrFile, (destinationPath).c_str(), TRUE);

		// If the file was copied successfully
		if (GetLastError() == 0)
		{
			LOG_INFO("Copied file: {} from file explorer to assets: {}", objFileName, destinationPath);

			return objFileName;
		}

		// If the file already exists
		if (GetLastError() == WIN32_API_ERROR_CODE_FILE_ALREADY_EXISTS)
		{
			// Assuming the file is the same one if the size is the same
			bool sameFileExists = fs::file_size(destinationPath) == objFileSize;
			if (sameFileExists)
			{
				std::string newDestinationPath = fs::current_path().string() + destinationPath + objFileName;
				LOG_INFO("File: {} already exists in assets: {}", objFileName, newDestinationPath);
				return objFileName;
			}

			LOG_TRACE("File name already exists! Trying again with number extension");

			// It will not recheck the size to determine if the file is the same after the initial check, not likely to happen but could fix it later
			for (int i = 1; i < 100; i++)
			{
				CopyFile(ofn.lpstrFile, (fs::current_path().string() + destinationPath + objFileNameNoExtension + std::to_string(i) + objFileExtension).c_str(), TRUE);

				objFileName = objFileNameNoExtension + std::to_string(i) + objFileExtension;

				if (GetLastError() != WIN32_API_ERROR_CODE_FILE_ALREADY_EXISTS)
				{
					break;
				}
			}
		}

		// Unknown error
		LOG_ERROR("Error copying file from file explorer to assets: " + destinationPath);
		return "";
	}
}