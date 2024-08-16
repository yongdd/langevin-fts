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
            0.6965456581, 0.636655225, 0.6514580668,
            0.5794545502, 0.6413949021, 0.5962758192,
            0.558548356, 0.6601148449, 0.5569728913,
            0.5964779091, 0.6290102494, 0.5775121486,
            0.5846974973, 0.6469315711, 0.6639138583,
            0.654692146, 0.5950073499, 0.6825497426,
            0.6917256734, 0.7245422629, 0.7022905036,
            0.6208944319, 0.7362918657, 0.6476201437,
            0.556910252, 0.651577934, 0.6122978018,
            0.5876833681, 0.6942208366, 0.616292124,
            0.5481693969, 0.7025850486, 0.6337584332,
            0.5391286738, 0.6224088075, 0.6143140535,
            0.5345032761, 0.5294697169, 0.520947629,
            0.5829711247, 0.6610041438, 0.5287456124,
            0.6601460967, 0.6659161313, 0.6197818348,
            0.5853524162, 0.5952154452, 0.6984995997,
            0.5638891268, 0.5313406813, 0.5343779299,
            0.6463252753, 0.5258684278, 0.5531855677,
            0.6586589231, 0.6413400744, 0.6505003159,
            0.7070963334, 0.6864069274, 0.6566075495,
        };
        double q2_last_ref[M] =
        {
            0.6810083246, 0.6042219428, 0.6088941863,
            0.5499790828, 0.5523265158, 0.6646200703,
            0.6104139336, 0.6635820753, 0.6213703022,
            0.6796826878, 0.7098425232, 0.6458523321,
            0.5548159682, 0.5798284317, 0.6281662988,
            0.5963987107, 0.6430736681, 0.6104627897,
            0.6593499107, 0.6631208324, 0.7252402836,
            0.6170169159, 0.7195208023, 0.6585338261,
            0.5794674771, 0.6725039984, 0.5752551656,
            0.6436001186, 0.642522178, 0.6871550254,
            0.5640114031, 0.670609007, 0.6181336276,
            0.5703167502, 0.6774451221, 0.6424661223,
            0.5786673846, 0.5496132976, 0.5417027025,
            0.5841556773, 0.5807653122, 0.5541754977,
            0.6424438503, 0.6198358109, 0.6386821682,
            0.5771929061, 0.5987387839, 0.6900534285,
            0.6009603513, 0.5254176256, 0.6024316286,
            0.628337461, 0.5247686088, 0.5741865074,
            0.6621998454, 0.7046183294, 0.598915981,
            0.6727811693, 0.6382628733, 0.5693589452,
        };

        double phi_a_ref[M] =
        {
            0.577096734097, 0.489952282739, 0.490513598895,
            0.426312959905, 0.444665900914, 0.546045882101,
            0.472858946316, 0.541573681757, 0.478679103461,
            0.53720684707, 0.572371519284, 0.508223652414,
            0.432705599669, 0.459773457402, 0.520762438388,
            0.48485501042, 0.517096333471, 0.492022692676,
            0.544789626328, 0.540181720625, 0.606824131529,
            0.488780655219, 0.607065958156, 0.526875535668,
            0.454843422129, 0.560394922015, 0.454079406495,
            0.517073718699, 0.527590694245, 0.565260798948,
            0.427220409253, 0.558714203606, 0.487738505277,
            0.441282446241, 0.547603292959, 0.518235355778,
            0.460457975023, 0.428889434168, 0.417483731816,
            0.464574552537, 0.482528463257, 0.429490977238,
            0.530179350417, 0.502357551073, 0.514025080434,
            0.451547841372, 0.473535335433, 0.589012354857,
            0.474815404023, 0.406884615733, 0.480825737461,
            0.525498071918, 0.400770791976, 0.452468099915,
            0.539723071105, 0.581898962755, 0.483170042066,
            0.564709948041, 0.525315372386, 0.452565790848,
        };
        double phi_b_ref[M] =
        {
            0.587259375812, 0.509225909147, 0.526498792537,
            0.444466746762, 0.510870361491, 0.488006589006,
            0.439135218267, 0.544603462393, 0.439314941111,
            0.481580040855, 0.515598168719, 0.460789219622,
            0.448730454405, 0.509485172276, 0.537419812202,
            0.525129975192, 0.466592523562, 0.548539111993,
            0.56751901373, 0.587618686469, 0.586113627082,
            0.493699662739, 0.617511357374, 0.522210303632,
            0.441829836166, 0.544125212444, 0.487762111098,
            0.475531740815, 0.568256863572, 0.50742326716,
            0.415537136492, 0.57859270342, 0.505525775932,
            0.421231055184, 0.508398145909, 0.496645616274,
            0.431592185595, 0.414582093297, 0.406224633043,
            0.46476248001, 0.540456959365, 0.412817898083,
            0.541368736652, 0.533750126308, 0.495227292423,
            0.458152342433, 0.470916675422, 0.594068728201,
            0.447555771588, 0.41079646005, 0.42913821037,
            0.541821551976, 0.400366410539, 0.442046008958,
            0.535113022812, 0.529208554684, 0.522016615325,
            0.590186520629, 0.562462774442, 0.516589956949,
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

        // Pseudo-spectral method
        #ifdef USE_CPU_MKL
        solver_name_list.push_back("pseudo, cpu-mkl");
        cb_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CpuComputationContinuous(cb_list.end()[-1], molecules, propagator_analyzer, "pseudospectral"));
        #endif

        #ifdef USE_CUDA
        solver_name_list.push_back("pseudo, cuda");
        solver_name_list.push_back("pseudo, cuda_reduce_memory_usage");
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CudaComputationContinuous(cb_list.end()[-2], molecules, propagator_analyzer, "pseudospectral"));
        solver_list.push_back(new CudaComputationReduceMemoryContinuous(cb_list.end()[-1], molecules, propagator_analyzer, "pseudospectral"));
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
            error = std::abs(QQ-14.899629822584/(Lx*Ly*Lz));
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

            error = std::abs(stress[0] + 0.000708638);
            std::cout<< "Stress[0] error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(stress[1] + 0.000935597);
            std::cout<< "Stress[1] error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(stress[2] + 0.00192901);
            std::cout<< "Stress[2] error: "<< error << std::endl;
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
