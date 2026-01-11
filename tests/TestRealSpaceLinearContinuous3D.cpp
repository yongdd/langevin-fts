#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <map>

#include "Exception.h"
#include "PropagatorComputationOptimizer.h"
#include "Molecules.h"
#include "Polymer.h"
#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#include "CpuComputationReduceMemoryContinuous.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#include "CudaComputationReduceMemoryContinuous.h"
#endif

typedef double T;

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
            7.3142865210e-01, 6.3988611300e-01, 6.5563943383e-01,
            5.5649940097e-01, 6.4361566149e-01, 6.0202070471e-01,
            5.5037939565e-01, 6.7239169905e-01, 5.4620658040e-01,
            5.8752115829e-01, 6.3132129548e-01, 5.7279683525e-01,
            5.6279010148e-01, 6.3289743696e-01, 6.6807979834e-01,
            6.5643490579e-01, 5.8175232139e-01, 6.7978708407e-01,
            7.0499875700e-01, 7.2735330058e-01, 7.2528820435e-01,
            6.1748670797e-01, 7.6079224362e-01, 6.4804719142e-01,
            5.4936409850e-01, 6.7355887408e-01, 6.0727404215e-01,
            5.9314124456e-01, 7.0627018425e-01, 6.3159524412e-01,
            5.1935481827e-01, 7.1780874947e-01, 6.2381641371e-01,
            5.3313704428e-01, 6.3021261699e-01, 6.1917445783e-01,
            5.3876409896e-01, 5.1641490140e-01, 5.0579175029e-01,
            5.7442074852e-01, 6.7332812691e-01, 5.2413285228e-01,
            6.7392638843e-01, 6.6070070572e-01, 6.0986259996e-01,
            5.7305173359e-01, 5.9245845387e-01, 7.3866529483e-01,
            5.5190082527e-01, 5.1951166705e-01, 5.3271256756e-01,
            6.8020075237e-01, 5.0742632258e-01, 5.5297394663e-01,
            6.5756983647e-01, 6.5184650316e-01, 6.5295583893e-01,
            7.3494083211e-01, 6.9603522625e-01, 6.4575769822e-01,
        };
        double q2_last_ref[M] =
        {
            7.1481020190e-01, 6.1223945063e-01, 6.0316226144e-01,
            5.2920880452e-01, 5.5104557383e-01, 6.7784411193e-01,
            5.9644233519e-01, 6.7034562568e-01, 5.9766684017e-01,
            6.5948448690e-01, 7.0502762497e-01, 6.3658937971e-01,
            5.4202782487e-01, 5.6618647672e-01, 6.4838875281e-01,
            5.9931984556e-01, 6.4834841742e-01, 6.0018699387e-01,
            6.7953313346e-01, 6.6637528844e-01, 7.6166200763e-01,
            6.1281231932e-01, 7.5114169438e-01, 6.5429755576e-01,
            5.6715698902e-01, 6.9794265062e-01, 5.6107042826e-01,
            6.4537009892e-01, 6.4657226392e-01, 7.0735578907e-01,
            5.3325803450e-01, 6.9069916678e-01, 5.9794218646e-01,
            5.5843011591e-01, 6.7954587383e-01, 6.4745025204e-01,
            5.8013377466e-01, 5.3755525777e-01, 5.2296726612e-01,
            5.7678900355e-01, 5.9802181724e-01, 5.4905025543e-01,
            6.5645205320e-01, 6.1699953737e-01, 6.3268905500e-01,
            5.6436577626e-01, 5.9702337243e-01, 7.3099826563e-01,
            5.8815552983e-01, 5.1419462130e-01, 6.0277899089e-01,
            6.6029715832e-01, 5.0918704576e-01, 5.6870186375e-01,
            6.6330853103e-01, 7.2230890564e-01, 6.0061488559e-01,
            7.0423628344e-01, 6.4993311911e-01, 5.6176921699e-01,
        };

        double phi_a_ref[M] =
        {
            6.0565725932e-01, 4.9514128901e-01, 4.8549256127e-01,
            4.0728856686e-01, 4.4642515247e-01, 5.5291488161e-01,
            4.6263614310e-01, 5.4847761847e-01, 4.5936950658e-01,
            5.1755986081e-01, 5.6447377357e-01, 5.0008296421e-01,
            4.2115548091e-01, 4.4918244858e-01, 5.3636078079e-01,
            4.8874592440e-01, 5.1498542445e-01, 4.8329952503e-01,
            5.6007615170e-01, 5.4007579160e-01, 6.3329863975e-01,
            4.8550822427e-01, 6.3051577408e-01, 5.2067559055e-01,
            4.4491038109e-01, 5.8024329497e-01, 4.4417505705e-01,
            5.1637292205e-01, 5.2845846623e-01, 5.7694658370e-01,
            4.0228993068e-01, 5.7555096571e-01, 4.7290920099e-01,
            4.3286567479e-01, 5.4449686727e-01, 5.1985636819e-01,
            4.6337117083e-01, 4.1805287577e-01, 4.0230633837e-01,
            4.5619861533e-01, 4.9924966684e-01, 4.2527474523e-01,
            5.4287615435e-01, 4.9760903579e-01, 5.0483972394e-01,
            4.3858740711e-01, 4.7098682168e-01, 6.2233105042e-01,
            4.5927705212e-01, 3.9801374324e-01, 4.7977110488e-01,
            5.5457730356e-01, 3.8690020205e-01, 4.4792489018e-01,
            5.3453100820e-01, 5.9032283749e-01, 4.8570672922e-01,
            5.9205418884e-01, 5.3476441536e-01, 4.4599787308e-01,
        };
        double phi_b_ref[M] =
        {
            6.1582646587e-01, 5.1006984500e-01, 5.2308733588e-01,
            4.2419516813e-01, 5.0987142888e-01, 4.9805701459e-01,
            4.3393379933e-01, 5.5222122137e-01, 4.2889673104e-01,
            4.7318589669e-01, 5.1721784626e-01, 4.6014251830e-01,
            4.3145430705e-01, 4.9358023185e-01, 5.4378678366e-01,
            5.2586983983e-01, 4.6174403262e-01, 5.3974423142e-01,
            5.7616167973e-01, 5.8168788473e-01, 6.0563507210e-01,
            4.8932692167e-01, 6.3665120288e-01, 5.1812135884e-01,
            4.3453117322e-01, 5.6218093559e-01, 4.7809771044e-01,
            4.8170866103e-01, 5.7148278124e-01, 5.2375183269e-01,
            3.9252158225e-01, 5.9110030177e-01, 4.9298134052e-01,
            4.1857419393e-01, 5.1283151551e-01, 5.0150923307e-01,
            4.3781732581e-01, 4.0442290529e-01, 3.9331220643e-01,
            4.5433752305e-01, 5.4844750090e-01, 4.0931640825e-01,
            5.5416038896e-01, 5.2618468739e-01, 4.8695902485e-01,
            4.4462979008e-01, 4.6710648352e-01, 6.2713411234e-01,
            4.3620897389e-01, 4.0324053534e-01, 4.3202137217e-01,
            5.6873237775e-01, 3.8397222951e-01, 4.3918906675e-01,
            5.3062912097e-01, 5.4031209125e-01, 5.2250138628e-01,
            6.1109243184e-01, 5.6530623365e-01, 5.0122574474e-01,
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
        PropagatorComputationOptimizer* propagator_computation_optimizer= new PropagatorComputationOptimizer(molecules, false);

        propagator_computation_optimizer->display_blocks();
        propagator_computation_optimizer->display_propagators();

        std::vector<PropagatorComputation<T>*> solver_list;
        std::vector<ComputationBox<T>*> cb_list;
        std::vector<std::string> solver_name_list;

        // Real space method
        #ifdef USE_CPU_MKL
        solver_name_list.push_back("real space, cpu-mkl");
        solver_name_list.push_back("real space, cpu-mkl, reduce_memory_usage");
        cb_list.push_back(new CpuComputationBox<T>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CpuComputationBox<T>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CpuComputationContinuous            <T>(cb_list.end()[-2], molecules, propagator_computation_optimizer, "realspace"));
        solver_list.push_back(new CpuComputationReduceMemoryContinuous<T>(cb_list.end()[-1], molecules, propagator_computation_optimizer, "realspace"));
        #endif

        #ifdef USE_CUDA
        solver_name_list.push_back("real space, cuda");
        solver_name_list.push_back("real space, cuda, reduce_memory_usage");
        cb_list.push_back(new CudaComputationBox<T>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox<T>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CudaComputationContinuous            <T>(cb_list.end()[-2], molecules, propagator_computation_optimizer, "realspace"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous<T>(cb_list.end()[-1], molecules, propagator_computation_optimizer, "realspace"));
        #endif

        // For each platform
        for(size_t n=0; n<solver_list.size(); n++)
        {
            PropagatorComputation<double>* solver = solver_list[n];
            ComputationBox<double>* cb = cb_list[n];

            for(int i=0; i<M; i++)
            {
                phi_a[i] = 0.0;
                phi_b[i] = 0.0;
                q1_last[i] = 0.0;
                q2_last[i] = 0.0;
            }

            //---------------- run --------------------
            std::cout<< std::endl << "Running: " << solver_name_list[n] << std::endl;

            const int p = 0;
            Polymer& pc = molecules->get_polymer(p);
            solver->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver->compute_concentrations();
            solver->get_total_concentration("A", phi_a);
            solver->get_total_concentration("B", phi_b);

            //--------------- check --------------------
            std::cout<< "Checking"<< std::endl;
            std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;

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
            error = std::abs(QQ-0.62209120738);
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
