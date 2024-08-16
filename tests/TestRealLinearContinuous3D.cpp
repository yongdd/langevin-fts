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
        const int NN{4};

        double q1_last[M], q2_last[M];

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, Lz, f;

        f = 0.5;
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

        double q1_last_ref[M] =
        {
            7.3426130033e-01, 6.3921173571e-01, 6.5541751740e-01, 
            5.5412219662e-01, 6.4505328774e-01, 6.0206511674e-01, 
            5.4948623437e-01, 6.7350397004e-01, 5.4457983014e-01, 
            5.8644373238e-01, 6.3056558414e-01, 5.7237654843e-01, 
            5.6096859298e-01, 6.3264076461e-01, 6.6910043949e-01, 
            6.5769953404e-01, 5.7922158587e-01, 6.7986247500e-01, 
            7.0615701201e-01, 7.2674703669e-01, 7.2604231189e-01, 
            6.1712059004e-01, 7.6265872264e-01, 6.4713882135e-01, 
            5.4882902837e-01, 6.7473729749e-01, 6.0717941147e-01, 
            5.9298318780e-01, 7.0644264732e-01, 6.3139106289e-01, 
            5.1676798140e-01, 7.1941733152e-01, 6.2384875759e-01, 
            5.3305457116e-01, 6.2938909496e-01, 6.1898721281e-01, 
            5.3979363381e-01, 5.1536805906e-01, 5.0467900286e-01, 
            5.7318215808e-01, 6.7578684370e-01, 5.2293834103e-01, 
            6.7606343504e-01, 6.6027265581e-01, 6.0863266325e-01, 
            5.7122500120e-01, 5.9146720377e-01, 7.4180234441e-01, 
            5.4997272717e-01, 5.1932471006e-01, 5.3265006764e-01, 
            6.8321290629e-01, 5.0478529614e-01, 5.5254239508e-01, 
            6.5655331005e-01, 6.5173755175e-01, 6.5403638815e-01, 
            7.3701973851e-01, 6.9688879028e-01, 6.4440309518e-01, 
        };
        double q2_last_ref[M] =
        {
            7.1758479342e-01, 6.1215484158e-01, 6.0108788633e-01, 
            5.2660532059e-01, 5.4981367173e-01, 6.8072012887e-01, 
            5.9617376347e-01, 6.7097121752e-01, 5.9604779366e-01, 
            6.5886932176e-01, 7.0509477097e-01, 6.3718168883e-01, 
            5.4063243505e-01, 5.6404473446e-01, 6.5060614501e-01, 
            5.9947301800e-01, 6.4975552820e-01, 5.9748909735e-01, 
            6.8034337962e-01, 6.6415880921e-01, 7.6399047005e-01, 
            6.1217270902e-01, 7.5311906263e-01, 6.5286669788e-01, 
            5.6671285160e-01, 7.0018448116e-01, 5.5906202087e-01, 
            6.4624459539e-01, 6.4466057030e-01, 7.0954084442e-01, 
            5.3119509129e-01, 6.9257720546e-01, 5.9642125359e-01, 
            5.5835408702e-01, 6.7917458176e-01, 6.4796181375e-01, 
            5.8166790894e-01, 5.3684464606e-01, 5.2163671271e-01, 
            5.7562656087e-01, 5.9920591004e-01, 5.4810625365e-01, 
            6.5825681366e-01, 6.1566524792e-01, 6.3268977507e-01, 
            5.6228911746e-01, 5.9633548164e-01, 7.3406928156e-01, 
            5.8661799775e-01, 5.1327929616e-01, 6.0475018861e-01, 
            6.6270773289e-01, 5.0711978326e-01, 5.6803114837e-01, 
            6.6236245277e-01, 7.2415450613e-01, 6.0018878864e-01, 
            7.0614995293e-01, 6.4998084237e-01, 5.5902776247e-01,
        };

        double phi_a_ref[M] =
        {
            6.0902314291e-01, 4.9506045620e-01, 4.8349199203e-01, 
            4.0458965621e-01, 4.4568411038e-01, 5.5573028211e-01, 
            4.6235571234e-01, 5.4942577232e-01, 4.5765090310e-01, 
            5.1675335598e-01, 5.6443311771e-01, 5.0069888222e-01, 
            4.1964938200e-01, 4.4723401504e-01, 5.3883150825e-01, 
            4.8923900839e-01, 5.1595779955e-01, 4.8084451914e-01, 
            5.6114301556e-01, 5.3789615001e-01, 6.3585293383e-01, 
            4.8493932900e-01, 6.3294658457e-01, 5.1919836332e-01, 
            4.4442860201e-01, 5.8276397378e-01, 4.4238141774e-01, 
            5.1720914359e-01, 5.2670364051e-01, 5.7908741348e-01, 
            4.0005086292e-01, 5.7784782341e-01, 4.7156240699e-01, 
            4.3288341207e-01, 5.4401156370e-01, 5.2036972859e-01, 
            4.6506323398e-01, 4.1725508297e-01, 4.0097112160e-01, 
            4.5487621985e-01, 5.0093924457e-01, 4.2434249028e-01, 
            5.4509216395e-01, 4.9629900430e-01, 5.0460047485e-01, 
            4.3635591489e-01, 4.7025869180e-01, 6.2603044960e-01, 
            4.5748117324e-01, 3.9730311442e-01, 4.8161252963e-01, 
            5.5762173672e-01, 3.8469264061e-01, 4.4725336139e-01, 
            5.3341949579e-01, 5.9214837407e-01, 4.8564509644e-01, 
            5.9445927575e-01, 5.3506346143e-01, 4.4328566692e-01,
        };
        double phi_b_ref[M] =
        {
            6.1922193814e-01, 5.0950828309e-01, 5.2247823716e-01, 
            4.2160372365e-01, 5.1102975309e-01, 4.9875513722e-01, 
            4.3323963181e-01, 5.5352445493e-01, 4.2726757397e-01, 
            4.7210330593e-01, 5.1661834573e-01, 4.6002704652e-01, 
            4.2964006013e-01, 4.9296769366e-01, 5.4532285963e-01, 
            5.2718179810e-01, 4.5992293225e-01, 5.3931751849e-01, 
            5.7747018934e-01, 5.8070302649e-01, 6.0694051423e-01, 
            4.8895684708e-01, 6.3895143075e-01, 5.1707857113e-01, 
            4.3401035099e-01, 5.6388417283e-01, 4.7766297507e-01, 
            4.8183177936e-01, 5.7129764408e-01, 5.2410329521e-01, 
            3.8995870734e-01, 5.9315863930e-01, 4.9276458741e-01, 
            4.1858898695e-01, 5.1207099753e-01, 5.0149030781e-01, 
            4.3915225601e-01, 4.0341285215e-01, 3.9215728751e-01, 
            4.5295373174e-01, 5.5106429578e-01, 4.0821486618e-01, 
            5.5660682547e-01, 5.2549315023e-01, 4.8583050732e-01, 
            4.4257200719e-01, 4.6616952273e-01, 6.3088016776e-01, 
            4.3419386931e-01, 4.0303139443e-01, 4.3245538358e-01, 
            5.7223125111e-01, 3.8135938577e-01, 4.3873702099e-01, 
            5.2945114769e-01, 5.4065967399e-01, 5.2350104809e-01, 
            6.1361049035e-01, 5.6617824059e-01, 4.9943030762e-01, 
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};
        std::vector<BlockInput> blocks = 
        {
            {"A",    f, 0, 1},
            {"B",1.0-f, 1, 2},
        };

        double phi_a[M]={0.0}, phi_b[M]={0.0};

        Molecules* molecules = new Molecules("Continuous", 1.0/NN, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});
        PropagatorAnalyzer* propagator_analyzer= new PropagatorAnalyzer(molecules, false);

        propagator_analyzer->display_blocks();
        propagator_analyzer->display_propagators();

        std::vector<PropagatorComputation*> solver_list;
        std::vector<ComputationBox*> cb_list;
        std::vector<std::string> solver_name_list;

        // Real space method
        #ifdef USE_CPU_MKL
        solver_name_list.push_back("real space, cpu-mkl");
        cb_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CpuComputationContinuous(cb_list.end()[-1], molecules, propagator_analyzer, "realspace"));
        #endif

        #ifdef USE_CUDA
        solver_name_list.push_back("real space, cuda");
        solver_name_list.push_back("real space, cuda_reduce_memory_usage");
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CudaComputationContinuous(cb_list.end()[-2], molecules, propagator_analyzer, "realspace"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(cb_list.end()[-1], molecules, propagator_analyzer, "realspace"));
        #endif

        // For each platform
        for(size_t n=0; n<solver_list.size(); n++)
        {
            PropagatorComputation* solver = solver_list[n];
            ComputationBox* cb = cb_list[n];

            for(int i=0; i<M; i++)
            {
                phi_a[i] = 0.0;
                phi_b[i] = 0.0;
                q1_last[i] = 0.0;
                q2_last[i] = 0.0;
            }

            //---------------- run --------------------
            std::cout<< std::endl << "Running: " << solver_name_list[n] << std::endl;
            solver->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver->compute_concentrations();
            solver->get_total_concentration("A", phi_a);
            solver->get_total_concentration("B", phi_b);

            //--------------- check --------------------
            std::cout<< "Checking"<< std::endl;
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
            
            const int p = 0;
            Polymer& pc = molecules->get_polymer(p);
            solver->get_chain_propagator(q1_last, p, 1, 2, pc.get_block(1,2).n_segment);
            for(int i=0; i<M; i++)
                diff_sq[i] = pow(q1_last[i] - q1_last_ref[i],2);

            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            solver->get_chain_propagator(q2_last, p, 1, 0, pc.get_block(1,0).n_segment);
            for(int i=0; i<M; i++)
                diff_sq[i] = pow(q2_last[i] - q2_last_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Complementary Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            double QQ = solver->get_total_partition(p);
            error = std::abs(QQ-0.621996847395786);
            std::cout<< "Total Propagator error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            if (solver->check_total_partition() == false)
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

            delete cb;
            delete solver;
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
