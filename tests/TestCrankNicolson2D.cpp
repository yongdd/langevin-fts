#include <iostream>
#include <cmath>
#include <algorithm>

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
        const int M{II*JJ};
        const int NN{4};

        double q_prev[M], q_next[M];

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, f;

        f = 0.5;
        Lx = 4.0;
        Ly = 3.0;

        double w_a[M] = {0.822383458999126,  0.180877073118435, 0.885692320279145,
                         0.89417010799111,   0.495074990864166, 0.612975741629382,
                         0.0415795198090432, 0.353431810889399, 0.773118461366249,
                         0.474587294381635,
                         0.200002276650706, 0.592127025743285, 0.460207620078036,
                         0.435945198378862, 0.61269588805607,  0.355979618324841,
                         0.548759176402544, 0.482897565408353, 0.541501788021353,
                         0.349682106604464};

        double w_b[M] = {0.104943438436657, 0.418086769592863, 0.61190542613784,
                         0.792961240687622, 0.713098832553561, 0.667410867822433,
                         0.492427261460169, 0.261956970404376, 0.479635996452285,
                         0.206215439022739,
                         0.180066106322814, 0.349989269840229, 0.580249533529743,
                         0.653060847246207, 0.793729416310673, 0.988032605316576,
                         0.98005550969782,  0.38678227079795,  0.894395839154923,
                         0.720491484453521};

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};
        std::vector<BlockInput> blocks = 
        {
            {"A",    f, 0, 1},
            {"B",1.0-f, 1, 2},
        };

        std::vector<std::string> bc_abs = 
        {
            "absorbing", "absorbing",
            "reflecting", "reflecting",
        };
        std::vector<std::string> bc_rfl = 
        {
            "reflecting", "reflecting",
            "periodic", "periodic",
        };
        std::vector<std::string> bc_prd = 
        {
            "periodic", "periodic",
            "absorbing", "absorbing",
        };

        Molecules* molecules = new Molecules("Continuous", 1.0/NN, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});
        PropagatorAnalyzer* propagator_analyzer= new PropagatorAnalyzer(molecules, false);

        propagator_analyzer->display_blocks();
        propagator_analyzer->display_propagators();

        std::vector<PropagatorComputation*> solver_list;
        std::vector<std::string> solver_name_list;
        
        int repeat = 0;

        #ifdef USE_CPU_MKL
        repeat += 1;
        solver_name_list.push_back("cpu-mkl, absorbing");
        solver_list.push_back(new CpuComputationContinuous(new CpuComputationBox({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_analyzer, "realspace"));
        #endif

        #ifdef USE_CUDA
        repeat += 2;
        solver_name_list.push_back("cuda, absorbing");
        solver_name_list.push_back("cuda_reduce_memory_usage, absorbing");
        solver_list.push_back(new CudaComputationContinuous(new CudaComputationBox({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_analyzer, "realspace"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(new CudaComputationBox({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_analyzer, "realspace"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, reflecting");
        solver_list.push_back(new CpuComputationContinuous(new CpuComputationBox({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_analyzer, "realspace"));
        #endif
        
        #ifdef USE_CUDA
        solver_name_list.push_back("cuda, reflecting");
        solver_name_list.push_back("cuda_reduce_memory_usage, reflecting");
        solver_list.push_back(new CudaComputationContinuous(new CudaComputationBox({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_analyzer, "realspace"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(new CudaComputationBox({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_analyzer, "realspace"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, periodic");
        solver_list.push_back(new CpuComputationContinuous(new CpuComputationBox({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_analyzer, "realspace"));
        #endif
        
        #ifdef USE_CUDA
        solver_name_list.push_back("cuda, periodic");
        solver_name_list.push_back("cuda_reduce_memory_usage, periodic");
        solver_list.push_back(new CudaComputationContinuous(new CudaComputationBox({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_analyzer, "realspace"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(new CudaComputationBox({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_analyzer, "realspace"));
        #endif

        // For each platform
        for(size_t n=0; n<solver_list.size(); n++)
        {
            PropagatorComputation* solver = solver_list[n];

            for(int i=0; i<M; i++)
                q_next[i] = 0.0;

            //---------------- run --------------------
            std::cout<< std::endl << "Running Pseudo: " << n << ", " << solver_name_list[n] << std::endl;
            solver->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver->compute_concentrations();

            solver->get_chain_propagator(q_next, 0, 1, 2, 1);
            if (n % repeat != 0)
            {
                //--------------- check --------------------
                std::cout<< "Checking"<< std::endl;
                std::cout<< "If error is less than 1.0e-7, it is ok!" << std::endl;
                std::cout<< "i: q (" << solver_name_list[n-1] << "), (" << solver_name_list[n] << ")" << std::endl;
                for(int i=0; i<M; i++)
                    std::cout << i << ": " << q_prev[i] << ", " << q_next[i] << std::endl;

                for(int i=0; i<M; i++)
                    diff_sq[i] = pow(q_prev[i] - q_next[i],2);
                error = sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));

                std::cout<< "Propagator error: "<< error << std::endl;
                if (!std::isfinite(error) || error > 1e-7)
                    return -1;
            }

            for(int i=0; i<M; i++)
                q_prev[i] = q_next[i];
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
