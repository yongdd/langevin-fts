/*----------------------------------------------------------
* class PlatformSelector
*-----------------------------------------------------------*/

#ifndef PLATFORM_SELECTOR_H_
#define PLATFORM_SELECTOR_H_

#include <string>
#include <vector>
#include "AbstractFactory.h"

class PlatformSelector
{
public:
    static AbstractFactory* create_factory(std::string str_platform="CUDA");
};

#endif
