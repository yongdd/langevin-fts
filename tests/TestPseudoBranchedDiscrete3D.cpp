#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>

#include "Exception.h"
#include "Polymer.h"
#include "PropagatorAnalyzer.h"
#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuComputationDiscrete.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationDiscrete.h"
#include "CudaComputationReduceMemoryDiscrete.h"
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
            2.6696168107e-03, 2.2237143379e-03, 2.2603693699e-03, 
            1.8332470791e-03, 1.8066857554e-03, 2.5643745167e-03, 
            3.0202476692e-03, 3.3286908832e-03, 3.1321477244e-03, 
            3.6722155905e-03, 3.9199037055e-03, 3.4549795163e-03, 
            1.9209340414e-03, 2.0651115545e-03, 2.2794732171e-03, 
            2.0673069286e-03, 2.3592232797e-03, 2.2118195217e-03, 
            3.0521518324e-03, 3.0535794246e-03, 3.7162420425e-03, 
            2.9287109039e-03, 3.6554080520e-03, 3.2446711016e-03, 
            2.1348699182e-03, 2.6494809930e-03, 2.0810409941e-03, 
            2.4722167946e-03, 2.3095759791e-03, 2.7765007008e-03, 
            2.2164846594e-03, 2.7644845574e-03, 2.5776320696e-03, 
            2.3442780402e-03, 3.0372494112e-03, 2.7778321190e-03, 
            1.9587158446e-03, 1.7579968614e-03, 1.7704531262e-03, 
            1.8699771968e-03, 1.7370413075e-03, 1.7964303644e-03, 
            2.2538740237e-03, 2.1042435411e-03, 2.2689685587e-03, 
            2.0017370116e-03, 2.1463839774e-03, 2.5610901767e-03, 
            2.0710715422e-03, 1.6502510292e-03, 2.1065131313e-03, 
            2.0900314164e-03, 1.6133395840e-03, 1.9040761042e-03, 
            2.6174642690e-03, 2.9698825716e-03, 2.2469761233e-03, 
            2.7536947512e-03, 2.5411754742e-03, 2.0811084483e-03, 
        };
        double q_1_0_last_ref[M] =
        {
            8.0801484740e-03, 6.6029444720e-03, 6.7526913278e-03, 
            5.1446756465e-03, 4.9977938643e-03, 7.3451179484e-03, 
            9.7192319626e-03, 1.0791634439e-02, 1.0067093165e-02, 
            1.2404491209e-02, 1.3275170613e-02, 1.1549080324e-02, 
            5.1793276584e-03, 5.6269251815e-03, 6.1511015839e-03, 
            5.2689897801e-03, 5.9670549323e-03, 5.7833928166e-03, 
            8.8054490539e-03, 8.8385360458e-03, 1.0959724888e-02, 
            8.8558907498e-03, 1.1234689761e-02, 9.8770185043e-03, 
            6.1467526844e-03, 7.7310718724e-03, 5.9664275218e-03, 
            6.7471741937e-03, 6.2235812715e-03, 7.6243008280e-03, 
            6.1280517855e-03, 7.6952758917e-03, 7.2806085974e-03, 
            6.8906133677e-03, 9.1329351994e-03, 8.3102592969e-03, 
            6.1706049499e-03, 5.4061979196e-03, 5.5209541269e-03, 
            5.5375200906e-03, 5.0032881650e-03, 5.2511568657e-03, 
            6.6219818253e-03, 6.1429073758e-03, 6.6871677851e-03, 
            6.1850624690e-03, 6.6010313457e-03, 8.0071738702e-03, 
            6.6296971604e-03, 5.1121612490e-03, 6.7117507479e-03, 
            6.4424773220e-03, 4.8554341373e-03, 5.8337539984e-03, 
            8.4111523598e-03, 9.6135568306e-03, 7.0857417333e-03, 
            9.1211118480e-03, 8.3469733900e-03, 6.7453796710e-03,
        };

        double phi_a_ref[M] =
        {
            6.3730142556e-01, 4.7790914171e-01, 4.9585150774e-01, 
            3.5426977573e-01, 3.4950054791e-01, 5.7540303133e-01, 
            7.1019282058e-01, 8.5491248478e-01, 7.5491514947e-01, 
            9.9137212919e-01, 1.1086599854e+00, 8.7983733678e-01, 
            3.7461496392e-01, 4.2780445076e-01, 4.9153950431e-01, 
            4.2254906700e-01, 5.0768863307e-01, 4.7535527399e-01, 
            7.4891555158e-01, 7.7438533566e-01, 1.0120066363e+00, 
            6.9059600610e-01, 1.0089810784e+00, 8.2276840694e-01, 
            4.1401127708e-01, 5.8981359044e-01, 4.0781891576e-01, 
            5.1914127069e-01, 4.9440832251e-01, 6.2023075379e-01, 
            4.3936004145e-01, 6.4153286073e-01, 5.6651864185e-01, 
            4.6944906262e-01, 7.2872681970e-01, 6.2686905729e-01, 
            3.6629081508e-01, 3.1488826665e-01, 3.1657883802e-01, 
            3.5305034811e-01, 3.2191197870e-01, 3.2137311421e-01, 
            4.7977051958e-01, 4.4085388311e-01, 4.8344176618e-01, 
            3.9667832412e-01, 4.3808833660e-01, 5.8751702973e-01, 
            4.2368134371e-01, 2.9577921543e-01, 4.2305656290e-01, 
            4.3018515052e-01, 2.8652066698e-01, 3.6557236605e-01, 
            6.1846706149e-01, 7.3920735251e-01, 4.8039858645e-01, 
            6.6826428354e-01, 5.9066169877e-01, 4.3602989421e-01, 
        };
        double phi_b_ref[M] =
        {
            5.3059591431e-01, 4.8454826509e-01, 4.8729923810e-01, 
            4.1545687566e-01, 4.7458570554e-01, 4.3024606849e-01, 
            4.5052050725e-01, 5.5546693432e-01, 4.4693085730e-01, 
            4.9746609381e-01, 5.4390939298e-01, 4.8068354152e-01, 
            4.5090887420e-01, 5.1909683093e-01, 5.2767373019e-01, 
            5.1949150317e-01, 4.7426376622e-01, 5.5071465906e-01, 
            6.1187677786e-01, 6.6483955471e-01, 6.3612396520e-01, 
            5.3479754812e-01, 6.7613662285e-01, 5.6936812433e-01, 
            3.4990411260e-01, 4.3408130365e-01, 3.9938291024e-01, 
            3.9258460401e-01, 4.8674460224e-01, 4.2239548579e-01, 
            3.9953378711e-01, 5.5192889585e-01, 4.8197195689e-01, 
            3.7581442530e-01, 4.6041001759e-01, 4.4535081086e-01, 
            2.8408888635e-01, 2.8559982054e-01, 2.7712450971e-01, 
            3.2655642981e-01, 3.8629642539e-01, 2.9447102462e-01, 
            4.3880739780e-01, 4.5182040294e-01, 4.0918263893e-01, 
            3.6547627068e-01, 3.8127417914e-01, 4.5306377598e-01, 
            3.3566552519e-01, 3.1695884876e-01, 3.1411179222e-01, 
            3.8751418479e-01, 3.1077555915e-01, 3.2119857517e-01, 
            4.6910317159e-01, 4.6567162449e-01, 4.6086891201e-01, 
            5.1724796696e-01, 5.0137969419e-01, 4.6915985937e-01, 
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.5}};
        // std::vector<std::string> block_monomer_types = {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"};
        // std::vector<double> contour_lengths = {0.6,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,1.2,1.2};
        // std::vector<int> v = {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13};
        // std::vector<int> u = {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18};

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

        Molecules* molecules = new Molecules("Discrete", 0.15, bond_lengths);
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
        solver_list.push_back(new CpuComputationDiscrete(cb_list.end()[-2], molecules, propagator_analyzer_1));
        solver_list.push_back(new CpuComputationDiscrete(cb_list.end()[-1], molecules, propagator_analyzer_2));
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
        solver_list.push_back(new CudaComputationDiscrete(cb_list.end()[-4], molecules, propagator_analyzer_1));
        solver_list.push_back(new CudaComputationDiscrete(cb_list.end()[-3], molecules, propagator_analyzer_2));
        solver_list.push_back(new CudaComputationReduceMemoryDiscrete(cb_list.end()[-2], molecules, propagator_analyzer_1));
        solver_list.push_back(new CudaComputationReduceMemoryDiscrete(cb_list.end()[-1], molecules, propagator_analyzer_2));
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

            double QQ = solver->get_total_partition(p);
            error = std::abs(QQ-1.5943694728e-03/(Lx*Ly*Lz));
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