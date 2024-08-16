#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#include "Exception.h"
#include "Polymer.h"
#include "PropagatorAnalyzer.h"
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
        
        double q_1_4_last[M]={0.0}, q_1_0_last[M]={0.0};

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, Lz;

        Lx = 4.0;
        Ly = 3.0;
        Lz = 2.0;

        // initialize pseudo spectral parameters
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

        double q_1_4_last_ref[M] =
        {
            2.5792560983e-03, 2.2312722072e-03, 2.2902702893e-03, 
            1.9002616092e-03, 1.8667855127e-03, 2.4550786309e-03, 
            2.9816355611e-03, 3.2431750713e-03, 3.0981526675e-03, 
            3.5536698572e-03, 3.7500137171e-03, 3.3514071697e-03, 
            1.9792692304e-03, 2.1324163519e-03, 2.2527775893e-03, 
            2.0898282585e-03, 2.3060447247e-03, 2.2597107485e-03, 
            2.9988317122e-03, 3.0575750379e-03, 3.5361161457e-03, 
            2.9229805040e-03, 3.5094567481e-03, 3.2075147245e-03, 
            2.1392337349e-03, 2.5499494907e-03, 2.1251575275e-03, 
            2.4031591114e-03, 2.3108401056e-03, 2.6414733565e-03, 
            2.2662508658e-03, 2.6969344000e-03, 2.5926841507e-03, 
            2.3467541148e-03, 2.9454358402e-03, 2.7142512189e-03, 
            1.9288120082e-03, 1.7793930912e-03, 1.8005101323e-03, 
            1.8853515564e-03, 1.7520748335e-03, 1.8128615631e-03, 
            2.2132450621e-03, 2.1227095911e-03, 2.2429396491e-03, 
            2.0386004428e-03, 2.1510678214e-03, 2.4580497516e-03, 
            2.0677010856e-03, 1.6931990161e-03, 2.0536903487e-03, 
            2.0428736451e-03, 1.6701410492e-03, 1.9118481742e-03, 
            2.5805251715e-03, 2.8352791996e-03, 2.2636680364e-03, 
            2.6807329597e-03, 2.5266037038e-03, 2.1560442718e-03,
        };
        double q_1_0_last_ref[M] =
        {
            7.8503181417e-03, 6.6877743330e-03, 6.8915862790e-03, 
            5.3973416052e-03, 5.2346000177e-03, 7.0757353621e-03, 
            9.6138251411e-03, 1.0523724372e-02, 9.9658582386e-03, 
            1.1975925417e-02, 1.2668001159e-02, 1.1197673190e-02, 
            5.4102145178e-03, 5.8823464730e-03, 6.1533060591e-03, 
            5.4037664300e-03, 5.9138707032e-03, 5.9786287986e-03, 
            8.7074260523e-03, 8.9078093834e-03, 1.0460215408e-02, 
            8.8818975602e-03, 1.0812753320e-02, 9.7937745559e-03, 
            6.1801892655e-03, 7.4505187234e-03, 6.1205980750e-03, 
            6.5886927407e-03, 6.2703035425e-03, 7.2786732814e-03, 
            6.3182106990e-03, 7.5573543368e-03, 7.3634809253e-03, 
            6.9273173807e-03, 8.8585179854e-03, 8.1346690924e-03, 
            6.0706557078e-03, 5.4840833612e-03, 5.6200564486e-03, 
            5.6011438340e-03, 5.0831140686e-03, 5.3245868708e-03, 
            6.5412821819e-03, 6.2428985584e-03, 6.6453016692e-03, 
            6.3218390602e-03, 6.6407566591e-03, 7.6876097167e-03, 
            6.6297904017e-03, 5.2828441318e-03, 6.5526980165e-03, 
            6.3166625966e-03, 5.0659205201e-03, 5.8782809948e-03, 
            8.3136169679e-03, 9.1897974887e-03, 7.1786706565e-03, 
            8.8942026395e-03, 8.3244616444e-03, 7.0259882844e-03,
        };

        double phi_a_ref[M] =
        {
            6.3467114341e-01, 4.7802291303e-01, 4.9852876927e-01, 
            3.5616614494e-01, 3.5122096343e-01, 5.6969359860e-01, 
            7.0899795563e-01, 8.5423928847e-01, 7.5559363865e-01, 
            9.9022172218e-01, 1.1060290440e+00, 8.7628391004e-01, 
            3.7601497056e-01, 4.3137466991e-01, 4.8957435042e-01, 
            4.2328989713e-01, 5.0369541289e-01, 4.7905790231e-01, 
            7.4916866474e-01, 7.7946564589e-01, 1.0075118431e+00, 
            6.9237238664e-01, 1.0073941097e+00, 8.2570667597e-01, 
            4.1463499034e-01, 5.8752961484e-01, 4.1078325481e-01, 
            5.1714664843e-01, 4.9697940896e-01, 6.1551573453e-01, 
            4.4141073580e-01, 6.4058322307e-01, 5.6973180434e-01, 
            4.6944800746e-01, 7.2894435321e-01, 6.2585245252e-01, 
            3.6468681648e-01, 3.1534953713e-01, 3.1748104201e-01, 
            3.5423491948e-01, 3.2230140569e-01, 3.2147693646e-01, 
            4.7886799925e-01, 4.4295407526e-01, 4.8337826445e-01, 
            3.9883385381e-01, 4.3880785964e-01, 5.8543016576e-01, 
            4.2411606547e-01, 2.9608882698e-01, 4.1984311995e-01, 
            4.2844411946e-01, 2.8771426336e-01, 3.6591981957e-01, 
            6.1926462265e-01, 7.3520085502e-01, 4.8137528712e-01, 
            6.6705998888e-01, 5.9208309153e-01, 4.3970947619e-01,
        };
        double phi_b_ref[M] =
        {
            5.2363739111e-01, 4.8376050742e-01, 4.8454872805e-01, 
            4.1907336325e-01, 4.6809964464e-01, 4.3286088387e-01, 
            4.5359818362e-01, 5.5228415123e-01, 4.5156448210e-01, 
            5.0322120031e-01, 5.4858923373e-01, 4.8450356566e-01, 
            4.5403298577e-01, 5.1674750333e-01, 5.2530900252e-01, 
            5.1473286381e-01, 4.8167300266e-01, 5.4656115707e-01, 
            6.0803408666e-01, 6.6195128115e-01, 6.3547812041e-01, 
            5.3525348704e-01, 6.7227741889e-01, 5.7116226735e-01, 
            3.5183843225e-01, 4.3256532545e-01, 3.9797233829e-01, 
            3.9462551930e-01, 4.8377056967e-01, 4.2540969612e-01, 
            4.0563563637e-01, 5.4786471771e-01, 4.8125308865e-01, 
            3.7776076462e-01, 4.6398151703e-01, 4.4744351546e-01, 
            2.8381784034e-01, 2.8795511312e-01, 2.7951975788e-01, 
            3.2841564235e-01, 3.8039412890e-01, 2.9620741025e-01, 
            4.3496970850e-01, 4.5133837299e-01, 4.1272843283e-01, 
            3.6843906167e-01, 3.8275752041e-01, 4.4768861550e-01, 
            3.3987249426e-01, 3.1770423687e-01, 3.1632111816e-01, 
            3.8110734647e-01, 3.1432956767e-01, 3.2169863618e-01, 
            4.7125766227e-01, 4.6834811947e-01, 4.5711017370e-01, 
            5.1080696433e-01, 4.9721348978e-01, 4.6744469265e-01,
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.5}};

        std::vector<BlockInput> blocks = 
        {
            {"A",0.6, 0, 1},
            {"A",1.2, 0, 2},
            {"B",1.2, 0, 5},
            {"B",0.9, 0, 6},
            {"A",0.9, 1, 4},
            {"A",1.2, 1,15},
            {"B",1.2, 2, 3},
            {"A",0.9, 2, 7},
            {"B",1.2, 2,10},
            {"B",1.2, 3,14},
            {"A",0.9, 4, 8},
            {"A",1.2, 4, 9},
            {"B",1.2, 7,19},
            {"A",0.9, 8,13},
            {"B",1.2, 9,12},
            {"A",1.2, 9,16},
            {"A",1.2,10,11},
            {"B",1.2,13,17},
            {"A",1.2,13,18},
        };

        double phi_a[M]={0.0}, phi_b[M]={0.0};

        Molecules* molecules = new Molecules("Continuous", 0.15, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});

        PropagatorAnalyzer* propagator_analyzer_1 = new PropagatorAnalyzer(molecules, false);
        propagator_analyzer_1->display_blocks();
        propagator_analyzer_1->display_propagators();

        PropagatorAnalyzer* propagator_analyzer_2 = new PropagatorAnalyzer(molecules, true);
        propagator_analyzer_2->display_blocks();
        propagator_analyzer_2->display_propagators();

        std::vector<PropagatorComputation*> solver_list;
        std::vector<ComputationBox*> cb_list;
        std::vector<std::string> solver_name_list;

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("pseudo, cpu-mkl");
        solver_name_list.push_back("pseudo, cpu-mkl, aggregated");
        cb_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CpuComputationContinuous(cb_list.end()[-2], molecules, propagator_analyzer_1, "pseudospectral"));
        solver_list.push_back(new CpuComputationContinuous(cb_list.end()[-1], molecules, propagator_analyzer_2, "pseudospectral"));
        #endif
        #ifdef USE_CUDA
        solver_name_list.push_back("pseudo, cuda");
        solver_name_list.push_back("pseudo, cuda, aggregated");
        solver_name_list.push_back("pseudo, cuda_reduce_memory_usage");
        solver_name_list.push_back("pseudo, cuda_reduce_memory_usage, aggregated");
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CudaComputationContinuous(cb_list.end()[-4], molecules, propagator_analyzer_1, "pseudospectral"));
        solver_list.push_back(new CudaComputationContinuous(cb_list.end()[-3], molecules, propagator_analyzer_2, "pseudospectral"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(cb_list.end()[-2], molecules, propagator_analyzer_1, "pseudospectral"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(cb_list.end()[-1], molecules, propagator_analyzer_2, "pseudospectral"));
        #endif

        std::vector<std::vector<double>> stress_list {{},{},{}};

        // For each platform    
        for(size_t n=0; n<solver_list.size(); n++)
        {
            PropagatorComputation* solver = solver_list[n];
            ComputationBox* cb = cb_list[n];

            for(int i=0; i<M; i++)
            {
                phi_a[i] = 0.0;
                phi_b[i] = 0.0;
                q_1_4_last[i] = 0.0;
                q_1_0_last[i] = 0.0;
            }

            //---------------- run --------------------
            std::cout<< std::endl << "Running: " << solver_name_list[n] << std::endl;
            solver->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver->compute_concentrations();
            solver->get_total_concentration("A", phi_a);
            solver->get_total_concentration("B", phi_b);
            //solver->get_block_concentration(0, phi);

            //--------------- check --------------------
            std::cout<< "Checking"<< std::endl;
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
            
            const int p = 0;
            Polymer& pc = molecules->get_polymer(p);
            solver->get_chain_propagator(q_1_4_last, p, 1, 4, pc.get_block(1,4).n_segment);
            for(int i=0; i<M; i++)
                diff_sq[i] = pow(q_1_4_last[i] - q_1_4_last_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            solver->get_chain_propagator(q_1_0_last, p, 1, 0, pc.get_block(1,0).n_segment);
            for(int i=0; i<M; i++)
                diff_sq[i] = pow(q_1_0_last[i] - q_1_0_last_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Complementary Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            if (solver->check_total_partition() == false)
                return -1;

            double QQ = solver->get_total_partition(p);
            error = std::abs(QQ-1.5701353236e-03/(Lx*Ly*Lz));
            std::cout<< "Total Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            for(int i=0; i<M; i++)
                diff_sq[i] = pow(phi_a[i] - phi_a_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Segment Concentration A error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            for(int i=0; i<M; i++)
                diff_sq[i] = pow(phi_b[i] - phi_b_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Segment Concentration B error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            solver->compute_stress();
            std::vector<double> stress = solver->get_stress();
            std::cout<< "Stress: " << stress[0] << ", " << stress[1] << ", " << stress[2] << std::endl;
            for(int i=0;i<3;i++)
                stress_list[i].push_back(stress[i]);

            delete cb;
            delete solver;
        }
        for(int i=0;i<3;i++)
        {
            double mean = std::accumulate(stress_list[i].begin(), stress_list[i].end(), 0.0)/stress_list[i].size();
            double sq_sum = std::inner_product(stress_list[i].begin(), stress_list[i].end(), stress_list[i].begin(), 0.0);
            double stddev = std::sqrt(std::abs(sq_sum / stress_list[i].size() - mean * mean));
            std::cout << "Std. of Stress[" + std::to_string(i) + "]: " << stddev << std::endl;
            if (stddev > 1e-7)
                return -1;
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
