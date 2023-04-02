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
    static std::vector<std::string> avail_platforms();
    static AbstractFactory* create_factory(std::string platform, std::string chain_model);
    static AbstractFactory* create_factory(std::string platform, std::string chain_model, bool reduce_memory_usage);
};

#endif
