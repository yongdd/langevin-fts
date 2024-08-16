#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>

#include "Exception.h"
#include "PropagatorAnalyzer.h"
#include "Molecules.h"
#include "Polymer.h"
#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#include "CudaComputationReduceMemoryContinuous.h"
#endif

int main()
{
    try
    {
        const int II{5};
        const int JJ{4};
        const int KK{3};
        const int M{II*JJ*KK};
        const int NN{100};

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, Lz, f;

        f = 0.5;
        Lx = 4.0;
        Ly = 3.0;
        Lz = 2.0;

        // Initialize pseudo spectral parameters
        double w_a[M] = {0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
                        0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
                        0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
                        0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
                        0.567929080e-1,0.568028857e+0,0.144986181e+0,0.466158777e+0,0.573327733e+0,
                        0.136324723e+0,0.819010407e+0,0.271218167e+0,0.626224101e+0,0.398109186e-1,
                        0.860031651e+0,0.338153865e+0,0.688078522e+0,0.564682952e+0,0.222924187e+0,
                        0.306816449e+0,0.316316038e+0,0.640568415e+0,0.702342408e+0,0.632135481e+0,
                        0.649402777e+0,0.647100865e+0,0.370402133e+0,0.691313864e+0,0.447870566e+0,
                        0.757298851e+0,0.586173682e+0,0.766745717e-1,0.504185402e+0,0.812016428e+0,
                        0.217988206e+0,0.273487202e+0,0.937672578e+0,0.570540523e+0,0.409071185e+0,
                        0.391548274e-1,0.663478965e+0,0.260755447e+0,0.503943226e+0,0.979481790e+0
                        };
        double w_b[M] = {0.113822903e-1,0.330673934e+0,0.270138412e+0,0.669606774e+0,0.885344778e-1,
                        0.604752856e+0,0.890062293e+0,0.328557615e+0,0.965824739e+0,0.865399960e+0,
                        0.698893686e+0,0.857947305e+0,0.594897904e+0,0.248187208e+0,0.155686710e+0,
                        0.116803898e+0,0.711146609e+0,0.107610460e+0,0.143034307e+0,0.123131521e+0,
                        0.230387237e+0,0.516274641e+0,0.562366089e-1,0.491449746e+0,0.746656140e+0,
                        0.296108614e+0,0.424987667e+0,0.651538750e+0,0.116745920e+0,0.567790110e+0,
                        0.954487190e+0,0.802476927e-1,0.440223916e+0,0.843025420e+0,0.612864528e+0,
                        0.571893767e+0,0.759625605e+0,0.872255004e+0,0.935065364e+0,0.635565347e+0,
                        0.373711972e-2,0.860683468e+0,0.186492706e+0,0.267880995e+0,0.579305501e+0,
                        0.693549226e+0,0.613843845e+0,0.259811620e-1,0.848915465e+0,0.766111508e+0,
                        0.872008750e+0,0.116289041e+0,0.917713893e+0,0.710076955e+0,0.442712526e+0,
                        0.516722213e+0,0.253395805e+0,0.472950065e-1,0.152934959e+0,0.292486174e+0
                        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};
        std::vector<BlockInput> blocks_1 = 
        {
            {"A",    f, 0, 1},
            {"B",1.0-f, 1, 2},
        };
        std::vector<BlockInput> blocks_2 = 
        {
            {"A", 0.01, 0, 1},
        };
        double phi_a[M]={0.0}, phi_solvent[M]={0.0};

        Molecules* molecules_1 = new Molecules("Continuous", 1.0/NN, bond_lengths);
        molecules_1->add_polymer(0.6, blocks_1, {});
        molecules_1->add_polymer(0.4, blocks_2, {});

        Molecules* molecules_2 = new Molecules("Continuous", 1.0/NN, bond_lengths);
        molecules_2->add_polymer(0.6, blocks_1, {});
        molecules_2->add_solvent(0.4, "A");

        PropagatorAnalyzer* propagator_analyzer_1 = new PropagatorAnalyzer(molecules_1, false);
        propagator_analyzer_1->display_blocks();
        propagator_analyzer_1->display_propagators();

        PropagatorAnalyzer* propagator_analyzer_2 = new PropagatorAnalyzer(molecules_2, false);
        propagator_analyzer_2->display_blocks();
        propagator_analyzer_2->display_propagators();

        std::vector<PropagatorComputation*> solver_1_list;
        std::vector<PropagatorComputation*> solver_2_list;
        std::vector<ComputationBox*> cb_1_list;
        std::vector<ComputationBox*> cb_2_list;
        std::vector<std::string> solver_name;

        #ifdef USE_CPU_MKL
        solver_name.push_back("CpuComputationContinuous, Aggregation=false");
        #endif
        #ifdef USE_CUDA
        solver_name.push_back("CudaComputationContinuous, Aggregation=false");
        solver_name.push_back("CudaComputationReduceMemoryContinuous, Aggregation=false");
        #endif

        #ifdef USE_CPU_MKL
        cb_1_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_1_list.push_back(new CpuComputationContinuous(cb_1_list.end()[-1], molecules_1, propagator_analyzer_1, "pseudospectral"));
        #endif
        #ifdef USE_CUDA
        cb_1_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_1_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_1_list.push_back(new CudaComputationContinuous(cb_1_list.end()[-2], molecules_1, propagator_analyzer_1, "pseudospectral"));
        solver_1_list.push_back(new CudaComputationReduceMemoryContinuous(cb_1_list.end()[-1], molecules_1, propagator_analyzer_1, "pseudospectral"));
        #endif

        #ifdef USE_CPU_MKL
        cb_2_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_2_list.push_back(new CpuComputationContinuous(cb_2_list.end()[-1], molecules_2, propagator_analyzer_2, "pseudospectral"));
        #endif
        #ifdef USE_CUDA
        cb_2_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_2_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_2_list.push_back(new CudaComputationContinuous(cb_2_list.end()[-2], molecules_2, propagator_analyzer_2, "pseudospectral"));
        solver_2_list.push_back(new CudaComputationReduceMemoryContinuous(cb_2_list.end()[-1], molecules_2, propagator_analyzer_2, "pseudospectral"));
        #endif

        // For each platform    
        for(size_t n=0; n<solver_1_list.size(); n++)
        {
            for(int i=0; i<M; i++)
            {
                phi_a[i] = 0.0;
                phi_solvent[i] = 0.0;
            }

            const int p = 1;
            const int s = 0;

            //---------------- run --------------------
            std::cout<< "Running Pseudo " << std::endl;
            solver_1_list[n]->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver_1_list[n]->compute_concentrations();
            solver_1_list[n]->get_total_concentration(p, "A", phi_a);

            solver_2_list[n]->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver_2_list[n]->compute_concentrations();
            solver_2_list[n]->get_solvent_concentration(s, phi_solvent);

            //--------------- check --------------------
            std::cout<< "Checking"<< std::endl;
            std::cout<< "If error is less than 1.0e-4, it is ok!" << std::endl;
            
            double QQ_1 = solver_1_list[n]->get_total_partition(p);
            double QQ_2 = solver_2_list[n]->get_solvent_partition(s);
            error = std::abs((QQ_1-QQ_2)/(Lx*Ly*Lz));
            std::cout<< "Total Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            for(int i=0; i<M; i++)
                diff_sq[i] = pow(phi_a[i] - phi_solvent[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Segment Concentration A error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-4)
                return -1;

            solver_1_list[n]->compute_stress();
            std::vector<double> stress_1 = solver_1_list[n]->get_stress();
            std::cout<< "Stress_1: " << stress_1[0] << ", " << stress_1[1] << ", " << stress_1[2] << std::endl;

            solver_2_list[n]->compute_stress();
            std::vector<double> stress_2 = solver_2_list[n]->get_stress();
            std::cout<< "Stress_2: " << stress_2[0] << ", " << stress_2[1] << ", " << stress_2[2] << std::endl;

            for(int i=0; i<3; i++)
            {
                error = std::abs(stress_1[i] - stress_2[i]);
                std::cout<< "Stress[" << std::to_string(i) << "] error: "<< error << std::endl;
                if (!std::isfinite(error) || error > 1e-7)
                    return -1;
            }

            delete cb_1_list[n];
            delete cb_2_list[n];
            delete solver_1_list[n];
            delete solver_2_list[n];
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
