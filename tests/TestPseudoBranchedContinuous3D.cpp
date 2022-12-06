#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>

#include "Exception.h"
#include "PolymerChain.h"
#include "BranchedPolymerChain.h"
#ifdef USE_CPU_MKL
#include "MklFFT3D.h"
#include "ComputationBox.h"
#include "CpuPseudoBranchedContinuous.h"
#endif
// #ifdef USE_CUDA
// #include "CudaComputationBox.h"
// #include "ComputationBox.h"
// #include "CudaPseudoContinuous.h"
// #endif

int main()
{
    try{
        const int II{5};
        const int JJ{4};
        const int KK{3};
        const int MM{II*JJ*KK};
        const int NN{10};

        double phi[MM*2];
        double q1_init[MM] {0.0}, q2_init[MM] {0.0};
        double q1_last[MM], q2_last[MM];

        std::array<double,MM> diff_sq;
        double QQ, error;
        double Lx, Ly, Lz, f;

        f = 0.5;
        Lx = 4.0;
        Ly = 3.0;
        Lz = 2.0;

        // initialize pseudo spectral parameters
        double w_a[MM] = {0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
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
        double w_b[MM] = {0.113822903e-1,0.330673934e+0,0.270138412e+0,0.669606774e+0,0.885344778e-1,
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
        std::vector<double> block_lengths = {f, 1.0-f};
        std::map<std::string, double> bond_length = {{"A",1.0}, {"B",1.0}};
        PolymerChain pc({"A","B"}, block_lengths, bond_length, 1.0/NN, "Continuous");
        BranchedPolymerChain bpc("Continuous", 0.1, {{"A",1.0}, {"B",1.0}},
        {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"},
        {0.4,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,1.2,1.2},
        {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13},
        {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18});

        std::vector<Pseudo*> pseudo_list;
        #ifdef USE_CPU_MKL
        pseudo_list.push_back(new CpuPseudoBranchedContinuous(new ComputationBox({II,JJ,KK}, {Lx,Ly,Lz}), &bpc, &pc, new MklFFT3D({II,JJ,KK})));
        #endif
        // #ifdef USE_CUDA
        // pseudo_list.push_back(new CudaPseudoContinuous(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}), &pc));
        // #endif

        // For each platform    
        for(Pseudo* pseudo : pseudo_list){
            for(int i=0; i<MM; i++){
                phi[i] = 0.0;
                phi[i+MM] = 0.0;
                q1_last[i] = 0.0;
                q2_last[i] = 0.0;
            }

            //---------------- run --------------------
            std::cout<< "Running Pseudo " << std::endl;
            //pseudo->compute_statistics(phi_a, phi_b, q1_init, q2_init, w_a, w_b, QQ);
            pseudo->compute_statistics(phi, q1_init, q2_init, {{"A",w_a},{"B",w_b}}, QQ);
        }

        //     //--------------- check --------------------
        //     std::cout<< "Checking"<< std::endl;
        //     std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
        //     pseudo->get_partition(q1_last, NN, q2_last, 0);

        //     for(int i=0; i<MM; i++)
        //         diff_sq[i] = pow(q1_last[i] - q1_last_ref[i],2);
        //     error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
        //     std::cout<< "Partial Partition error: "<< error << std::endl;
        //     if (std::isnan(error) || error > 1e-7)
        //         return -1;

        //     for(int i=0; i<MM; i++)
        //         diff_sq[i] = pow(q2_last[i] - q2_last_ref[i],2);
        //     error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
        //     std::cout<< "Complementary Partial Partition error: "<< error << std::endl;
        //     if (std::isnan(error) || error > 1e-7)
        //         return -1;


        //     for(int i=0; i<MM; i++)
        //         diff_sq[i] = pow(phi[i] - phi_a_ref[i],2);
        //     error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
        //     std::cout<< "Segment Concentration A error: "<< error << std::endl;
        //     if (std::isnan(error) || error > 1e-7)
        //         return -1;

        //     for(int i=0; i<MM; i++)
        //         diff_sq[i] = pow(phi[i+MM] - phi_b_ref[i],2);
        //     error = sqrt(*std::max_element(diff_sq.begin(),diff_sq.end()));
        //     std::cout<< "Segment Concentration B error: "<< error << std::endl;
        //     if (std::isnan(error) || error > 1e-7)
        //         return -1;
            
        //     delete pseudo;
        // }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
