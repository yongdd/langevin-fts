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
private:
    std::vector<std::string> valid_strings;
    std::string str_platform;

    void init_valid_strings();
public:
    PlatformSelector();
    PlatformSelector(std::string str_platform);
    AbstractFactory* create_factory();
};

#endif
