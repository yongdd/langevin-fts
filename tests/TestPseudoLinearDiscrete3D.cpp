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
        const int N{4};

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
            0.7367350528, 0.6383551887, 0.6640303501,
            0.5548709899, 0.6622359659, 0.5896396590,
            0.5392164056, 0.6789652926, 0.5344618666,
            0.5788143117, 0.6210023978, 0.5608699333,
            0.5624839379, 0.6497811287, 0.6756957020,
            0.6699629164, 0.5630901669, 0.6984988419,
            0.7131781687, 0.7410617359, 0.7209973121,
            0.6168056570, 0.7691281634, 0.6450473512,
            0.5456860283, 0.6718343139, 0.6175541966,
            0.5799095532, 0.7154660484, 0.6125663369,
            0.5120497073, 0.7277216840, 0.6347692672,
            0.5225484893, 0.6168004831, 0.6102178679,
            0.5333946766, 0.5112347930, 0.5019666471,
            0.5739629441, 0.6955870868, 0.5110340920,
            0.6812569936, 0.6699093834, 0.6058659400,
            0.5683292336, 0.5853939526, 0.7427041536,
            0.5431904793, 0.5162905904, 0.5210808715,
            0.6854703239, 0.4966183127, 0.5473753956,
            0.6565188163, 0.6397814783, 0.6643697285,
            0.7451651245, 0.7103777775, 0.6601950486,
        };
        double q2_last_ref[M] =
        {
            0.7176798338, 0.6042915134, 0.5918953965,
            0.5214327604, 0.5336551113, 0.7050641313,
            0.6030734875, 0.6727616933, 0.6054058803,
            0.6821619404, 0.7273732214, 0.6530054747,
            0.5337739222, 0.5525011785, 0.6486222303,
            0.5922360463, 0.6657859925, 0.5870320613,
            0.6703921730, 0.6483386868, 0.7642618613,
            0.6066312205, 0.7522239521, 0.6506764194,
            0.5695715197, 0.7048235407, 0.5501111800,
            0.6591493852, 0.6355455687, 0.7269872291,
            0.5340271472, 0.6919529358, 0.5968680314,
            0.5603798854, 0.6893958389, 0.6536639371,
            0.5884808671, 0.5384218655, 0.5224917997,
            0.5732989974, 0.5848726347, 0.5420588551,
            0.6599956151, 0.6105407189, 0.6433096595,
            0.5553995223, 0.5903447374, 0.7330459344,
            0.5944356298, 0.5084134611, 0.6216544457,
            0.6532703899, 0.4981837103, 0.5649959528,
            0.6659624911, 0.7423527556, 0.5904088681,
            0.6964611731, 0.6392740127, 0.5386998310,
        };

        double phi_a_ref[M] =
        {
            0.5871562287, 0.4892126961, 0.4803222522,
            0.4165447787, 0.4353225376, 0.5636010020,
            0.4743275961, 0.5435223875, 0.4752130936,
            0.5393121200, 0.5789326444, 0.5152844972,
            0.4263677521, 0.4471423120, 0.5277012144,
            0.4813512541, 0.5294900686, 0.4785851601,
            0.5459291645, 0.5291016062, 0.6198496049,
            0.4853039923, 0.6154492655, 0.5218506812,
            0.4524953967, 0.5709268689, 0.4430186335,
            0.5244194242, 0.5194095251, 0.5817059590,
            0.4197088306, 0.5642942524, 0.4784799618,
            0.4414051368, 0.5508317383, 0.5232616336,
            0.4667437362, 0.4267941422, 0.4127462707,
            0.4597221118, 0.4807482055, 0.4279820455,
            0.5351420776, 0.4958254468, 0.5155217805,
            0.4435361286, 0.4713285209, 0.6005676834,
            0.4727973264, 0.4032498536, 0.4916780023,
            0.5323528604, 0.3938073723, 0.4495651820,
            0.5381591404, 0.5957594220, 0.4787623392,
            0.5699577844, 0.5218485135, 0.4385727829,
        };
        double phi_b_ref[M] =
        {
            0.5993807185, 0.5098300867, 0.5316831637,
            0.4376979637, 0.5225050305, 0.4821473007,
            0.4330566920, 0.5498296049, 0.4311081613,
            0.4720934145, 0.5086061011, 0.4539093918,
            0.4432387553, 0.5137131000, 0.5409635302,
            0.5328488535, 0.4534261939, 0.5563070814,
            0.5739998238, 0.5931699259, 0.5883305477,
            0.4930671480, 0.6252618748, 0.5200700575,
            0.4378866885, 0.5479433910, 0.4916055413,
            0.4710763477, 0.5749977914, 0.5017635200,
            0.4049011981, 0.5862626098, 0.5073617443,
            0.4171306125, 0.5028988123, 0.4931058415,
            0.4312110044, 0.4086039642, 0.4004673077,
            0.4605693900, 0.5543776337, 0.4080195352,
            0.5487134895, 0.5349700570, 0.4881430580,
            0.4525277350, 0.4680060097, 0.6065352658,
            0.4382451057, 0.4084425123, 0.4231915324,
            0.5553229703, 0.3921807190, 0.4400985039,
            0.5308744790, 0.5240701503, 0.5287361831,
            0.6026298306, 0.5704206516, 0.5204642918,
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

        Molecules* molecules = new Molecules("Discrete", 1.0/N, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});
        PropagatorAnalyzer* propagator_analyzer= new PropagatorAnalyzer(molecules, false);

        propagator_analyzer->display_blocks();
        propagator_analyzer->display_propagators();

        std::vector<PropagatorComputation*> solver_list;
        std::vector<ComputationBox*> cb_list;
        std::vector<std::string> solver_name_list;

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl");
        cb_list.push_back(new CpuComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CpuComputationDiscrete(cb_list.end()[-1], molecules, propagator_analyzer));
        #endif
        
        #ifdef USE_CUDA
        solver_name_list.push_back("cuda");
        solver_name_list.push_back("cuda_reduce_memory_usage");
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list.push_back(new CudaComputationBox({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CudaComputationDiscrete(cb_list.end()[-2], molecules, propagator_analyzer));
        solver_list.push_back(new CudaComputationReduceMemoryDiscrete(cb_list.end()[-1], molecules, propagator_analyzer));
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
            std::cout<< std::endl << "Running Pseudo: " << solver_name_list[n] << std::endl;
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
            error = std::abs(QQ-14.9276505263205/(Lx*Ly*Lz));
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

            error = std::abs(stress[0] + 0.000473764);
            std::cout<< "Stress[0] error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(stress[1] + 0.0006085);
            std::cout<< "Stress[1] error: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;

            error = std::abs(stress[2] + 0.00148826);
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
