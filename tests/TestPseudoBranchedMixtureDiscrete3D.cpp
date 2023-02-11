#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "PolymerChain.h"
#ifdef USE_CPU_MKL
#include "MklFFT3D.h"
#include "CpuPseudoDiscrete.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaPseudoDiscrete.h"
#endif

int main()
{
    try
    {
        const int II{5};
        const int JJ{4};
        const int KK{3};
        const int MM{II*JJ*KK};

        std::array<double,MM> diff_sq;
        double error;
        double Lx, Ly, Lz;

        Lx = 4.0;
        Ly = 3.0;
        Lz = 2.0;

        // initialize pseudo spectral parameters
        double w_a[MM] = { 2.4653017345e-01,-8.4924926185e-01, 6.4079998942e-01,
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

        double w_b[MM] = { 1.6991139622e-01,-2.1409489979e-01, 2.9605393528e-02,
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

        double w_c[MM] = {2.2102067421e-01,-4.2083846494e-01, 9.6704214711e-01,
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

        double phi_a_ref[MM] =
        {
            1.4461712482e-01, 2.8417396570e-01, 8.8747404262e-02, 
            1.1757650760e-01, 1.3490313029e-01, 2.1454887257e-01, 
            1.5720318547e-01, 4.1264591112e-01, 2.1617900514e-01, 
            2.4816794201e-01, 3.9933129079e-01, 1.2546574794e-01, 
            6.3847842443e-01, 3.7087064415e-01, 2.1525220828e-01, 
            4.7688693222e-01, 5.4375843956e-01, 2.7461675958e-01, 
            3.5315962248e-01, 3.6873129673e-01, 7.5756011503e-01, 
            4.1107104484e-01, 5.3811607518e-01, 3.3640138694e-01, 
            7.4815358959e-01, 3.2162016897e-01, 1.0549028191e+00, 
            4.3994807831e-01, 1.1820552999e+00, 8.7056695158e-01, 
            3.4434952304e-01, 9.3849770834e-01, 1.0741733250e+00, 
            3.4499444020e-01, 4.3816133929e-01, 1.0588898385e+00, 
            1.3340116686e-01, 2.6349153812e-01, 4.2641465826e-01, 
            3.4206038719e-01, 3.3171639504e-01, 3.3916033317e-01, 
            6.3142374473e-01, 5.0522975114e-01, 8.8480698027e-01, 
            2.2142199732e-01, 1.6729072525e-01, 2.8905688099e-01, 
            4.8723027080e-02, 7.2438167299e-02, 7.3880571116e-02, 
            1.1103080676e-01, 1.5549004432e-01, 2.5054985301e-01, 
            1.9904890912e-01, 2.9992987732e-01, 4.8318549161e-01, 
            1.3780340805e-01, 1.0631788556e-01, 2.3650002276e-01, 
        };
        double phi_b_ref[MM] =
        {
            3.7523182274e-01, 4.0884044910e-01, 3.7825772443e-01, 
            4.4495530889e-01, 3.2368951118e-01, 4.1867780102e-01, 
            3.8617835112e-01, 4.1821123353e-01, 4.9131744486e-01, 
            3.8478088108e-01, 4.8563151952e-01, 4.7683841293e-01, 
            5.7351552153e-01, 5.4673982638e-01, 5.4077111583e-01, 
            4.1632622071e-01, 4.2805435741e-01, 3.6727863860e-01, 
            3.6341636011e-01, 3.5875851017e-01, 4.9067768230e-01, 
            4.7318636365e-01, 5.1780931388e-01, 4.6959616541e-01, 
            4.2291249183e-01, 3.7410052481e-01, 4.0395190691e-01, 
            5.4311335001e-01, 5.0702196075e-01, 4.7835708985e-01, 
            6.1173931584e-01, 5.8343064640e-01, 5.3364173523e-01, 
            5.0418463434e-01, 3.8822352311e-01, 4.9434638718e-01, 
            4.0878240367e-01, 3.7807159055e-01, 3.8338695076e-01, 
            5.0609869949e-01, 4.2970586421e-01, 4.4006968918e-01, 
            4.5962462093e-01, 3.8687932594e-01, 4.5896818221e-01, 
            2.8252921012e-01, 2.6040830736e-01, 2.9811783820e-01, 
            3.2452320342e-01, 3.0692271928e-01, 2.9790827478e-01, 
            4.6813827409e-01, 3.8400335756e-01, 3.7102707098e-01, 
            4.2993984873e-01, 4.2921226552e-01, 3.3213314378e-01, 
            3.1692706596e-01, 3.0401925918e-01, 2.5932908779e-01, 
        };
        double phi_c_ref[MM] =
        {
            1.3421219574e-01, 1.9217074633e-01, 1.1356329204e-01, 
            1.8699166901e-01, 2.2877463570e-01, 1.4949885282e-01, 
            2.1773901167e-01, 2.2124043395e-01, 1.5334572276e-01, 
            1.4567686225e-01, 2.4814470213e-01, 1.6221161040e-01, 
            2.4591979214e-01, 2.5576213033e-01, 2.5541006930e-01, 
            2.3755579260e-01, 3.2104346034e-01, 2.7797603640e-01, 
            2.2819116940e-01, 1.8620205795e-01, 3.1208487627e-01, 
            1.7882228405e-01, 2.5692577992e-01, 2.1882415017e-01, 
            2.2506586023e-01, 2.0983017556e-01, 2.2883618451e-01, 
            2.9854136116e-01, 3.9180807882e-01, 3.6771702735e-01, 
            2.9013232229e-01, 2.2046391358e-01, 2.8502493572e-01, 
            1.4407781345e-01, 1.5228163549e-01, 2.8807076451e-01, 
            1.0318948385e-01, 9.4408275392e-02, 1.1241688406e-01, 
            1.8370117492e-01, 1.9987930027e-01, 1.9381720848e-01, 
            1.5450982779e-01, 1.8458112038e-01, 1.8788535099e-01, 
            9.6804664628e-02, 7.2491761590e-02, 9.9349216072e-02, 
            7.6714582272e-02, 8.0910447650e-02, 1.0980379926e-01, 
            1.6892376879e-01, 1.4353117153e-01, 1.5784472476e-01, 
            1.4815535103e-01, 1.3062150194e-01, 1.1797784104e-01, 
            9.0422365298e-02, 8.3484833575e-02, 1.2279884035e-01, 
        };

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",2.0}, {"C",1.5}};

        std::vector<double> volume_fraction;
        std::vector<std::vector<std::string>> block_monomer_types;
        std::vector<std::vector<double>> contour_lengths;
        std::vector<std::vector<int>> v;
        std::vector<std::vector<int>> u;

        volume_fraction.push_back(0.5);
        block_monomer_types.push_back({"C","A","A","A","A","A","C","A","A","A","A","A","C","A","A","A","A","A","C","A","A","A","A","A","C","B","B","B","B","B","C","B","B","B","B","B","C","B","B","B","B","B","C","B","B","B","B","B"});
        contour_lengths.push_back({1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9,1.2,0.9,0.9,0.9,0.9,0.9});
        v.push_back({0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7});
        u.push_back({1,9,10,11,12,13,2,14,15,16,17,18,3,19,20,21,22,23,4,24,25,26,27,28,5,29,30,31,32,33,6,34,35,36,37,38,7,39,40,41,42,43,8,44,45,46,47,48});

        volume_fraction.push_back(0.3);
        block_monomer_types.push_back({"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"});
        contour_lengths.push_back({0.6,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,1.2,1.2});
        v.push_back({0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13});
        u.push_back({1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18});

        volume_fraction.push_back(0.2);
        block_monomer_types.push_back({"A","A","B","B","C","C","B","A","C","C","A","B","C","C","C","C","B","B","B","B","C","B","B","C","C","B","B","C","B"});
        contour_lengths.push_back({0.9,1.2,1.2,0.9,0.9,1.2,1.2,0.9,0.9,1.2,1.2,1.2,1.2,0.9,1.2,1.2,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,1.2,0.9,0.9,0.9});
        v.push_back({0,0,0,0,1,1,2,2,2,3,3,5,5,8,8,8,8,9,10,11,13,14,14,16,16,18,21,22,23});
        u.push_back({1,2,8,12,4,5,3,11,13,6,7,9,18,10,14,20,26,27,15,17,16,19,22,21,29,23,24,25,28});

        double phi_a[MM]={0.0}, phi_b[MM]={0.0}, phi_c[MM]={0.0};

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

        Mixture* mx1 = new Mixture("Discrete", 0.15, bond_lengths, false);
        for(size_t p=0; p<block_monomer_types.size(); p++){
            mx1->add_polymer(volume_fraction[p], block_monomer_types[p], contour_lengths[p], v[p], u[p], {});
            std::cout << "block size: " << block_monomer_types[p].size() << std::endl;
        }
        mx1->display_unique_blocks();
        mx1->display_unique_branches();

        Mixture* mx2 = new Mixture("Discrete", 0.15, bond_lengths, true);
        for(size_t p=0; p<block_monomer_types.size(); p++){
            mx2->add_polymer(volume_fraction[p], block_monomer_types[p], contour_lengths[p], v[p], u[p], {});
            std::cout << "block size: " << block_monomer_types[p].size() << std::endl;
        }
        mx2->display_unique_blocks();
        mx2->display_unique_branches();

        std::vector<Pseudo*> pseudo_list;
        #ifdef USE_CPU_MKL
        pseudo_list.push_back(new CpuPseudoDiscrete(new ComputationBox({II,JJ,KK}, {Lx,Ly,Lz}), mx1, new MklFFT3D({II,JJ,KK})));
        pseudo_list.push_back(new CpuPseudoDiscrete(new ComputationBox({II,JJ,KK}, {Lx,Ly,Lz}), mx2, new MklFFT3D({II,JJ,KK})));
        #endif
        #ifdef USE_CUDA
        pseudo_list.push_back(new CudaPseudoDiscrete(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}), mx1));
        pseudo_list.push_back(new CudaPseudoDiscrete(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}), mx2));
        #endif

        std::vector<std::vector<int>> stress_list {{},{},{}};

        // For each platform    
        for(Pseudo* pseudo : pseudo_list)
        {
            for(int i=0; i<MM; i++)
            {
                phi_a[i] = 0.0;
                phi_b[i] = 0.0;
                phi_c[i] = 0.0;
            }

            //---------------- run --------------------
            std::cout<< "Running Pseudo " << std::endl;
            pseudo->compute_statistics({}, {{"A",w_a},{"B",w_b}, {"C",w_c}});
            pseudo->get_monomer_concentration("A", phi_a);
            pseudo->get_monomer_concentration("B", phi_b);
            pseudo->get_monomer_concentration("C", phi_c);

            //--------------- check --------------------
            std::cout<< "Checking"<< std::endl;
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;

            for(int p=0; p<mx1->get_n_polymers();p++)
                std::cout<< std::setprecision(10) << std::scientific << "Total Partial Partition (" + std::to_string(p) + "): " << pseudo->get_total_partition(p) << std::endl;

            error = std::abs(pseudo->get_total_partition(0)-1.3999194661e+05/(Lx*Ly*Lz))/std::abs(pseudo->get_total_partition(0));
            std::cout<< "Total Partial Partition (0) error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(pseudo->get_total_partition(1)-1.5625863384e+03/(Lx*Ly*Lz))/std::abs(pseudo->get_total_partition(1));
            std::cout<< "Total Partial Partition (1) error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(pseudo->get_total_partition(2)-1.6167175694e+03/(Lx*Ly*Lz))/std::abs(pseudo->get_total_partition(2));
            std::cout<< "Total Partial Partition (2) error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            for(int i=0; i<MM; i++)
                diff_sq[i] = pow(phi_a[i] - phi_a_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Segment Concentration A error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            for(int i=0; i<MM; i++)
                diff_sq[i] = pow(phi_b[i] - phi_b_ref[i],2);
            error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
            std::cout<< "Segment Concentration B error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            for(int i=0; i<MM; i++)
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

            std::vector<double> stress = pseudo->compute_stress();
            std::cout<< "Stress: " << stress[0] << ", " << stress[1] << ", " << stress[2] << std::endl;
            for(int i=0;i<3;i++)
                stress_list[i].push_back(stress[i]);

            delete pseudo;
        }
        for(int i=0;i<3;i++)
        {
            double mean = std::accumulate(stress_list[i].begin(), stress_list[i].end(), 0.0)/stress_list[i].size();
            double sq_sum = std::inner_product(stress_list[i].begin(), stress_list[i].end(), stress_list[i].begin(), 0.0);
            double stddev = std::sqrt(sq_sum / stress_list[i].size() - mean * mean);
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
