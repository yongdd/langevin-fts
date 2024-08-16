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

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, Lz;

        Lx = 4.0;
        Ly = 3.0;
        Lz = 2.0;

        // initialize pseudo spectral parameters
        double w_a[M] = { 2.4653017345e-01,-8.4924926185e-01, 6.4079998942e-01,
                        4.5189857495e-01, 8.1530724190e-01,-6.1719453339e-01,
                        4.8956544855e-01,-8.8248220720e-01, 3.0581985487e-01,
                        -4.5380053533e-01,-5.4676694151e-01, 7.5098234290e-01,
                        -7.8746803471e-01, 4.4725330718e-02, 7.0788601437e-01,
                        -5.1033604406e-01,-5.7904212261e-01, 7.6116351873e-01,
                        -1.5416470322e-01, 4.3392219781e-01,-9.3625385975e-01,
                        -2.7528617739e-01,-6.5623801575e-01, 3.4553088283e-01,
                        -8.3419364519e-01, 9.0912433069e-01,-9.4931057035e-01,
                        4.5884701488e-01,-9.5771026055e-01,-4.8861989189e-01,
                        6.2670877480e-01,-6.8576342263e-01,-6.3252238150e-01,
                        3.8299085203e-01,-2.2886823729e-01,-9.1367800841e-01,
                        9.8000309241e-01,-6.9715978249e-01,-9.2746201151e-01,
                        -3.1159798893e-01, 2.3047896665e-01, 4.8491924625e-01,
                        -7.7377019396e-01,-3.2557245361e-01,-9.3837828475e-01,
                        -1.0269347502e-01, 5.3193987318e-01, 4.7989332744e-01,
                        8.0404031670e-01, 5.1132430735e-01, 7.2489155265e-01,
                        4.1069028010e-01,-5.4440975803e-02,-5.4894485930e-01,
                        3.2165699731e-01,-3.6738814654e-01,-7.9590178990e-01,
                        -1.0435627716e-01, 7.4952608270e-01,-7.4492707076e-01,
                        };

        double w_b[M] = { 1.6991139622e-01,-2.1409489979e-01, 2.9605393528e-02,
                        -7.1234107205e-01, 9.1946237297e-01,-4.8180715350e-01,
                        2.1215587811e-01,-1.6048890869e-01,-9.6393356175e-01,
                        1.1590024768e-01,-7.1886124208e-01,-8.8643800835e-01,
                        -9.3288750750e-01,-6.7766996964e-01,-8.0825611273e-01,
                        2.7015139508e-01, 1.6518368071e-02, 9.6693218810e-01,
                        8.6826063739e-01, 9.8905046652e-01,-5.3505231866e-01,
                        -1.1060508976e-01,-4.9843847632e-01, 1.8247469110e-01,
                        2.4832810151e-01, 6.0041491127e-01, 4.1899660766e-01,
                        -4.8678142302e-01,-1.5396615447e-01, 5.2379887561e-02,
                        -9.9035043789e-01,-9.2900117658e-01,-1.8254716456e-01,
                        -7.7765006554e-01, 4.4753934574e-01,-5.1826897112e-01,
                        -8.0045382649e-01,-6.3647984340e-01,-5.3694914125e-01,
                        -5.6529273056e-01, 4.1472728096e-02,-7.1193777712e-02,
                        -3.8054785825e-01, 2.8351751536e-01,-5.7510051601e-01,
                        8.1312535346e-01, 9.2623330966e-01, 4.5786209116e-01,
                        -1.3253226242e-01, 2.3002684434e-02, 1.6215261208e-01,
                        -8.9753051279e-01,-1.6396722304e-01, 5.0129064640e-02,
                        -6.3754987835e-01,-8.1242642308e-01, 6.0531041741e-01,
                        -2.6763206691e-01, 3.8419380198e-02, 8.4290069518e-01,
                        };

        double w_c[M] = {2.2102067421e-01,-4.2083846494e-01, 9.6704214711e-01,
                        -2.5554658013e-01,-9.6188979038e-01, 3.7062134463e-01,
                        -7.9767624877e-01,-3.8815527859e-01, 6.8122337424e-01,
                        3.4514350929e-01,-9.6855585557e-01,-9.7153081596e-02,
                        -1.7865124885e-01,-2.8274111996e-02,-5.8350621328e-01,
                        1.7749012431e-01,-8.5242137305e-01,-4.3128129804e-01,
                        -2.5419578991e-01, 8.7054086810e-01,-8.4690355928e-01,
                        5.0996822850e-01,-6.1528174407e-01, 1.4310548266e-01,
                        -2.1643805927e-01,-7.3551235502e-02, 5.0716101147e-01,
                        -2.0991487968e-01,-7.5654103973e-01,-7.5645980056e-01,
                        -8.3897856449e-01, 7.0014174781e-01, 2.8198318766e-01,
                        9.1933712679e-01, 3.8530509331e-01,-9.5066245469e-01,
                        3.1831932769e-01, 5.5442386930e-01, 4.4703655997e-01,
                        -4.1009591408e-03,-2.8483076454e-01,-8.5928572096e-02,
                        5.9744416736e-01,-4.6211501234e-01, 5.2607489613e-02,
                        -4.4880915887e-02, 9.0939369392e-01, 6.0869995384e-01,
                        8.6410772044e-01, 6.7201114714e-01,-4.0647266217e-01,
                        -5.3674527770e-01,-2.2421053103e-02,-4.8118931728e-01,
                        -1.4469233609e-01, 3.5828043072e-01, 8.3716045446e-01,
                        1.7180116069e-01, 6.3570650780e-01,-8.0810537828e-01,
                        };

        double phi_a_ref[M] =
        {
            1.4588864076e-01, 2.8199135627e-01, 8.9483878128e-02, 
            1.1825193966e-01, 1.3711593226e-01, 2.1202269625e-01, 
            1.5855218603e-01, 4.0845945878e-01, 2.1822413804e-01, 
            2.4709320986e-01, 4.0172929037e-01, 1.2721515267e-01, 
            6.3793110902e-01, 3.7329278488e-01, 2.1735231682e-01, 
            4.7376941893e-01, 5.4070923543e-01, 2.7869799643e-01, 
            3.5179712697e-01, 3.7250347732e-01, 7.5276986518e-01, 
            4.1114485903e-01, 5.3673658401e-01, 3.3990934986e-01, 
            7.4084635280e-01, 3.2759715518e-01, 1.0436355902e+00, 
            4.4545923982e-01, 1.1770971549e+00, 8.6912498507e-01, 
            3.4897470246e-01, 9.3704226295e-01, 1.0730183883e+00, 
            3.4874593349e-01, 4.3649076515e-01, 1.0570243902e+00, 
            1.3616009059e-01, 2.5982029496e-01, 4.2017902708e-01, 
            3.4097423284e-01, 3.3512435935e-01, 3.4453173508e-01, 
            6.3052070637e-01, 5.0719194865e-01, 8.8314810861e-01, 
            2.2019823354e-01, 1.6862206386e-01, 2.9365064562e-01, 
            4.9063658692e-02, 7.3041927451e-02, 7.4973821356e-02, 
            1.1222824796e-01, 1.5566921339e-01, 2.4972301126e-01, 
            2.0149797025e-01, 3.0071433311e-01, 4.8071153091e-01, 
            1.3787441002e-01, 1.0795940427e-01, 2.3387084273e-01, 
        };
        double phi_b_ref[M] =
        {
            3.8366972539e-01, 4.0845209883e-01, 3.8272648304e-01, 
            4.3155217145e-01, 3.3968289897e-01, 4.0943124816e-01, 
            3.9422087762e-01, 4.1754264331e-01, 4.7518327273e-01, 
            3.8997391168e-01, 4.8241125077e-01, 4.6341406680e-01, 
            5.6033866206e-01, 5.4097414943e-01, 5.2391268239e-01, 
            4.2049699942e-01, 4.2290285319e-01, 3.8578384115e-01, 
            3.7523629052e-01, 3.7342640564e-01, 4.7398368174e-01, 
            4.7499429298e-01, 5.1072282367e-01, 4.8457968767e-01, 
            4.3345928466e-01, 3.8194645617e-01, 4.1425517921e-01, 
            5.3606235862e-01, 5.1162893922e-01, 4.8023415228e-01, 
            5.9940866517e-01, 5.6373678018e-01, 5.4258382097e-01, 
            4.8778015765e-01, 3.9909628187e-01, 4.8635831474e-01, 
            3.9694839109e-01, 3.6793749512e-01, 3.7612362318e-01, 
            5.0800493866e-01, 4.3839542038e-01, 4.4625406396e-01, 
            4.5879715310e-01, 3.9625887367e-01, 4.4826781963e-01, 
            2.9532485137e-01, 2.6908173769e-01, 3.0242263953e-01, 
            3.2646756997e-01, 3.0847773604e-01, 3.0086934383e-01, 
            4.6130101921e-01, 3.8590445387e-01, 3.7739252687e-01, 
            4.2477714259e-01, 4.1639606299e-01, 3.4644704284e-01, 
            3.1125372413e-01, 3.0514211994e-01, 2.7008119732e-01, 
        };
        double phi_c_ref[M] =
        {
            1.3409454004e-01, 1.9208463484e-01, 1.1744044236e-01, 
            1.8800684739e-01, 2.2253958008e-01, 1.5142608489e-01, 
            2.1261045115e-01, 2.2096819335e-01, 1.5764211477e-01, 
            1.4795804089e-01, 2.4494750339e-01, 1.6214046392e-01, 
            2.4719128519e-01, 2.6054976214e-01, 2.5142444240e-01, 
            2.4108402371e-01, 3.1490197658e-01, 2.7592016102e-01, 
            2.2546542127e-01, 1.9102747112e-01, 3.0555619844e-01, 
            1.8137226571e-01, 2.5263089899e-01, 2.2076768301e-01, 
            2.2283573794e-01, 2.0842780630e-01, 2.3570270608e-01, 
            3.0082972266e-01, 3.8975986118e-01, 3.6098808569e-01, 
            2.8208712524e-01, 2.2655296721e-01, 2.9141965260e-01, 
            1.4855934481e-01, 1.5211313433e-01, 2.7914342547e-01, 
            1.0418273815e-01, 9.5699074947e-02, 1.1388620467e-01, 
            1.8519982466e-01, 1.9965061456e-01, 1.9434229854e-01, 
            1.5904516849e-01, 1.8102298489e-01, 1.8896788043e-01, 
            9.4688708192e-02, 7.3378230546e-02, 1.0042634236e-01, 
            7.8891880313e-02, 8.2323168488e-02, 1.0855300721e-01, 
            1.6873145376e-01, 1.4478960278e-01, 1.5586712559e-01, 
            1.4822479941e-01, 1.3288716984e-01, 1.2115223606e-01, 
            9.0584644406e-02, 8.4848654382e-02, 1.1884703344e-01, 
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",2.0}, {"C",1.5}};

        std::vector<double> volume_fraction;
        std::vector<std::vector<BlockInput>> block_inputs;

        volume_fraction.push_back(0.5);
        std::vector<BlockInput> block_input_1 = 
        {
            {"C", 1.2, 0, 1},
            {"A", 0.9, 0, 9},
            {"A", 0.9, 0,10},
            {"A", 0.9, 0,11},
            {"A", 0.9, 0,12},
            {"A", 0.9, 0,13},
            {"C", 1.2, 1, 2},
            {"A", 0.9, 1,14},
            {"A", 0.9, 1,15},
            {"A", 0.9, 1,16},
            {"A", 0.9, 1,17},
            {"A", 0.9, 1,18},
            {"C", 1.2, 2, 3},
            {"A", 0.9, 2,19},
            {"A", 0.9, 2,20},
            {"A", 0.9, 2,21},
            {"A", 0.9, 2,22},
            {"A", 0.9, 2,23},
            {"C", 1.2, 3, 4},
            {"A", 0.9, 3,24},
            {"A", 0.9, 3,25},
            {"A", 0.9, 3,26},
            {"A", 0.9, 3,27},
            {"A", 0.9, 3,28},
            {"C", 1.2, 4, 5},
            {"B", 0.9, 4,29},
            {"B", 0.9, 4,30},
            {"B", 0.9, 4,31},
            {"B", 0.9, 4,32},
            {"B", 0.9, 4,33},
            {"C", 1.2, 5, 6},
            {"B", 0.9, 5,34},
            {"B", 0.9, 5,35},
            {"B", 0.9, 5,36},
            {"B", 0.9, 5,37},
            {"B", 0.9, 5,38},
            {"C", 1.2, 6, 7},
            {"B", 0.9, 6,39},
            {"B", 0.9, 6,40},
            {"B", 0.9, 6,41},
            {"B", 0.9, 6,42},
            {"B", 0.9, 6,43},
            {"C", 1.2, 7, 8},
            {"B", 0.9, 7,44},
            {"B", 0.9, 7,45},
            {"B", 0.9, 7,46},
            {"B", 0.9, 7,47},
            {"B", 0.9, 7,48},
        };

        volume_fraction.push_back(0.3);
        std::vector<BlockInput> block_input_2 = 
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

        volume_fraction.push_back(0.2);
        std::vector<BlockInput> block_input_3 = 
        {
            {"A", 0.9, 0, 1},
            {"A", 1.2, 0, 2},
            {"B", 1.2, 0, 8},
            {"B", 0.9, 0,12},
            {"C", 0.9, 1, 4},
            {"C", 1.2, 1, 5},
            {"B", 1.2, 2, 3},
            {"A", 0.9, 2,11},
            {"C", 0.9, 2,13},
            {"C", 1.2, 3, 6},
            {"A", 1.2, 3, 7},
            {"B", 1.2, 5, 9},
            {"C", 1.2, 5,18},
            {"C", 0.9, 8,10},
            {"C", 1.2, 8,14},
            {"C", 1.2, 8,20},
            {"B", 1.2, 8,26},
            {"B", 1.2, 9,27},
            {"B", 0.9,10,15},
            {"B", 1.2,11,17},
            {"C", 1.2,13,16},
            {"B", 0.9,14,19},
            {"B", 1.2,14,22},
            {"C", 1.2,16,21},
            {"C", 1.2,16,29},
            {"B", 1.2,18,23},
            {"B", 0.9,21,24},
            {"C", 0.9,22,25},
            {"B", 0.9,23,28},
        };

        block_inputs.push_back(block_input_1);
        block_inputs.push_back(block_input_2);
        block_inputs.push_back(block_input_3);

        double phi_a[M]={0.0}, phi_b[M]={0.0}, phi_c[M]={0.0};

        // for(int k=0; k<KK; k++)
        // {
        //     for(int j=0; j<JJ; j++)
        //     {
        //         for(int i=0; i<II; i++)
        //         {
        //             std::cout << std::setw(16) << std::setprecision(8) << std::scientific << w_a[i*JJ*KK + j*KK + k] << " " << w_b[i*JJ*KK + j*KK + k] <<  " " << w_c[i*JJ*KK + j*KK + k] << std::endl; 
        //         }
        //     }
        // }

        Molecules* molecules = new Molecules("Continuous", 0.15, bond_lengths);
        for(size_t p=0; p<block_inputs.size(); p++){
            molecules->add_polymer(volume_fraction[p], block_inputs[p], {});
            std::cout << "block size: " << block_inputs[p].size() << std::endl;
        }

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
                phi_c[i] = 0.0;
            }

            //---------------- run --------------------
            std::cout<< std::endl << "Running: " << solver_name_list[n] << std::endl;
            solver->compute_propagators({{"A",w_a},{"B",w_b}, {"C",w_c}},{});
            solver->compute_concentrations();
            solver->get_total_concentration("A", phi_a);
            solver->get_total_concentration("B", phi_b);
            solver->get_total_concentration("C", phi_c);

            //--------------- check --------------------
            std::cout<< "Checking"<< std::endl;
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;

            for(int p=0; p<molecules->get_n_polymer_types();p++)
                std::cout<< std::setprecision(10) << std::scientific << "Total Partial Partition (" + std::to_string(p) + "): " << solver->get_total_partition(p) << std::endl;

            error = std::abs(solver->get_total_partition(0)-1.1763668568e+05/(Lx*Ly*Lz))/std::abs(solver->get_total_partition(0));
            std::cout<< "Total Partial Partition (0) error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(solver->get_total_partition(1)-1.4384268278e+03/(Lx*Ly*Lz))/std::abs(solver->get_total_partition(1));
            std::cout<< "Total Partial Partition (1) error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(solver->get_total_partition(2)-1.3821533787e+03/(Lx*Ly*Lz))/std::abs(solver->get_total_partition(2));
            std::cout<< "Total Partial Partition (2) error: "<< error << std::endl;
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

            for(int i=0; i<M; i++)
                diff_sq[i] = pow(phi_c[i] - phi_c_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Segment Concentration C error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            // for(int k=0; k<KK; k++)
            // {
            //     for(int j=0; j<JJ; j++)
            //     {
            //         for(int i=0; i<II; i++)
            //         {
            //             std::cout<< std::setprecision(10) << std::scientific << phi_a[i*JJ*KK + j*KK + k] << " " << phi_b[i*JJ*KK + j*KK + k] << " " << phi_c[i*JJ*KK + j*KK + k] << std::endl;
            //         }
            //     }
            // }
            
            // for(int i=0; i<II*JJ*KK; i++)
            // {
            //     std::cout<< std::setprecision(10) << std::scientific << phi_a[i] << ", ";
            //     if (i % 3 == 2)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl;

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
