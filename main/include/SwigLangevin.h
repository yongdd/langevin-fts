
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>

#include "ParamParser.h"
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"


Class LangevinAPI
{
private:
    // max_saddle_iter = maximum number of saddle point approximation iteration steps
    int saddle_max_iter;
    
public:

void find_saddle_point()
{
/*
    for(int saddle_iter=0; iter<saddle_max_iter; iter++)
    {
        // for the given fields find the polymer statistics
        pseudo->find_phi(phia, phib,q1_init,q2_init,&w[0],&w[sb->get_MM()],QQ);

        for(int i=0; i<sb->get_MM(); i++)
        {
            // calculate pressure field for the new field calculation, the method is modified from Fredrickson's
            xi[i] = 0.5*(w[i]+w[i+sb->get_MM()]-pc->get_chi_n());
            // calculate output fields
            w_out[i]              = pc->get_chi_n()*phib[i] + xi[i];
            w_out[i+sb->get_MM()] = pc->get_chi_n()*phia[i] + xi[i];
        }
        sb->zero_mean(&w_out[0]);
        sb->zero_mean(&w_out[sb->get_MM()]);

        // error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level;
        for(int i=0; i<2*sb->get_MM(); i++)
            w_diff[i] = w_out[i]- w[i];
        error_level = sqrt(sb->multi_inner_product(2,w_diff,w_diff)/
                           (sb->multi_inner_product(2,w,w)+1.0));

        // conditions to end the iteration
        if(error_level < tolerance) break;
        // calculte new fields using simple and Anderson mixing
        am->caculate_new_fields(w, w_out, w_diff, old_error_level, error_level);
    }
*/
}
}
