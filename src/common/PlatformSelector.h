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
    static AbstractFactory* create_factory(std::string platform);
    static AbstractFactory* create_factory(std::string platform, bool reduce_memory_usage);
    static AbstractFactory* create_factory(std::string platform, std::string data_type);
    static AbstractFactory* create_factory(std::string platform, std::string data_type, bool reduce_memory_usage);
};

#endif
