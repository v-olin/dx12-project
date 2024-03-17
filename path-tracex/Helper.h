#pragma once

#include "PathWin.h"

#include <exception>

#define THROW_IF_FAILED(hrcall) if(FAILED(hr = (hrcall))) { throw std::exception(); }
