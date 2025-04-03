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
    static AbstractFactory<double>* create_factory_real(std::string platform, bool reduce_memory_usage);
    static AbstractFactory<std::complex<double>>* create_factory_complex(std::string platform, bool reduce_memory_usage);

};

#endif
