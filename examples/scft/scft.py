import os
import numpy as np
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

class SCFT:
    def __init__(self, params):

        # choose platform among [cuda, cpu-mkl]
        avail_platforms = PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1: # for 1D simulation, use CPU
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms: # If cuda is available, use GPU
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        distinct_polymers = []
        assert(len(params['segment_lengths']) == 2), \
            "Currently, only AB copolymers are supported."
        assert(len(set(["A","B"]).intersection(set(params['segment_lengths'].keys())))==2), \
            "Use letters 'A' and 'B' for two species."
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        # (c++ class) Create a factory for given platform and chain_model
        factory = PlatformSelector.create_factory(platform, params["chain_model"])

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Polymer chains
        total_volume_fraction = 0.0
        for polymer in params["distinct_polymers"]:
            block_length_list = []
            type_list = []
            A_fraction = 0.0
            alpha = 0.0  #total_relative_contour_length
            for block in polymer["blocks"]:
                block_length_list.append(block["length"])
                type_list.append(block["type"])
                alpha += block["length"]
                if block["type"] == "A":
                    A_fraction += block["length"]
                elif block["type"] == "random":
                    A_fraction += block["length"]*block["fraction"]["A"]
            total_volume_fraction += polymer["volume_fraction"]

            total_A_fraction = A_fraction/alpha
            statistical_segment_length = \
                np.sqrt(params["segment_lengths"]["A"]**2*total_A_fraction + \
                        params["segment_lengths"]["B"]**2*(1-total_A_fraction))

            if "random" in set(bt.lower() for bt in type_list):
                assert(len(type_list) == 1), \
                    "Currently, Only single block random copolymer is supported."
                assert(np.isclose(polymer["blocks"][0]["fraction"]["A"]+polymer["blocks"][0]["fraction"]["B"],1.0)), \
                    "The sum of volume fraction of random copolymer must be equal to 1."
                segment_length_list = {"random":statistical_segment_length}
            else:
                segment_length_list = params["segment_lengths"]
            
            # (C++ class) Polymer chain
            pc = factory.create_polymer_chain(type_list, block_length_list, segment_length_list, params["ds"])

            # (C++ class) Solvers using Pseudo-spectral method
            pseudo = factory.create_pseudo(cb, pc)

            distinct_polymers.append(
                {"volume_fraction":polymer["volume_fraction"],
                 "block_types":type_list,
                 "total_A_fraction":total_A_fraction,
                 "statistical_segment_length":statistical_segment_length,
                 "alpha":alpha, "pc":pc, "pseudo":pseudo, })

        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["box_is_altering"] : 
            am_n_var = 2*np.prod(params["nx"]) + len(params["lx"])
        else :
            am_n_var = 2*np.prod(params["nx"])
        if "am" in params :
            am = factory.create_anderson_mixing(am_n_var,
                params["am"]["max_hist"],     # maximum number of history
                params["am"]["start_error"],  # when switch to AM from simple mixing
                params["am"]["mix_min"],      # minimum mixing rate of simple mixing
                params["am"]["mix_init"])     # initial mixing rate of simple mixing
        else : 
            am = factory.create_anderson_mixing(am_n_var, 20, 1e-2, 0.1,  0.1)

       # The maximum iteration steps
        if "max_iter" in params :
            max_iter = params["max_iter"]
        else :
            max_iter = 2000      # the number of maximum iterations

        # Tolerance
        if "tolerance" in params :
            tolerance = params["tolerance"]
        else :
            tolerance = 1e-8     # Terminate iteration if the self-consistency error is less than tolerance

        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
        print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
        print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
        print("Volume: %f" % (cb.get_volume()) )
        
        print("%s chain model" % (params["chain_model"]))
        print("chi_n: %f," % (params["chi_n"]))
        print("Conformational asymmetry (epsilon): %f" %
            (params["segment_lengths"]["A"]/params["segment_lengths"]["B"]))
        idx = 0
        for polymer in distinct_polymers:
            print("distinct_polymers[%d]:" % (idx) )
            print("    volume fraction: %f, alpha: %f, N: %d" %
                (polymer["volume_fraction"], polymer["alpha"], polymer["pc"].get_n_segment_total()), end=",")
            print(" sequence of block types:", polymer["block_types"])
            print("    total A fraction: %f, average statistical segment length: %f" % 
                (polymer["total_A_fraction"], polymer["statistical_segment_length"]))
            idx += 1

        #  Save Internal Variables
        self.distinct_polymers = distinct_polymers
        self.chi_n = params["chi_n"]
        self.box_is_altering = params["box_is_altering"]
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.cb = cb
        self.am = am

    def run(self, initial_fields):

        # assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # reset Anderson mixing module
        self.am.reset_count()

        # array for concentrations
        phi = {"A":np.zeros([self.cb.get_n_grid()], dtype=np.float64),
               "B":np.zeros([self.cb.get_n_grid()], dtype=np.float64)}

        # array for output fields
        w_out = np.zeros([2, self.cb.get_n_grid()], dtype=np.float64)

        # array for the initial condition
        # free end initial condition. q[0,:] is q and q[1,:] is qdagger.
        # q starts from one end and qdagger starts from the other end.
        q1_init = np.ones(self.cb.get_n_grid(), dtype=np.float64)
        q2_init = np.ones(self.cb.get_n_grid(), dtype=np.float64)

        #------------------ run ----------------------
        print("---------- Run ----------")

        # iteration begins here
        if (self.box_is_altering):
            print("iteration, mass error, total_partitions, energy_total, error_level, box size")
        else:
            print("iteration, mass error, total_partitions, energy_total, error_level")

        # reshape initial fields
        w = np.reshape([initial_fields["A"], initial_fields["B"]], [2, self.cb.get_n_grid()])

        # keep the level of field value
        self.cb.zero_mean(w[0])
        self.cb.zero_mean(w[1])

        for scft_iter in range(1, self.max_iter+1):
            # for the given fields find the polymer statistics
            phi["A"][:] = 0.0
            phi["B"][:] = 0.0
            for polymer in self.distinct_polymers:
                frac_ = polymer["volume_fraction"]/polymer["alpha"]
                if not "random" in set(polymer["block_types"]):
                    phi_, Q_ = polymer["pseudo"].compute_statistics(q1_init,q2_init, {"A":w[0],"B":w[1]})
                    for i in range(len(polymer["block_types"])):
                        phi[polymer["block_types"][i]] += frac_*phi_[i]
                elif set(polymer["block_types"]) == set(["random"]):
                    phi_, Q_ = polymer["pseudo"].compute_statistics(q1_init,q2_init, {"random":w[0]*polymer["total_A_fraction"] + w[1]*(1.0-polymer["total_A_fraction"])})
                    phi["A"] += frac_*phi_[0]*polymer["total_A_fraction"]
                    phi["B"] += frac_*phi_[0]*(1.0-polymer["total_A_fraction"])
                else:
                    raise ValueError("Unknown species,", set(polymer["block_types"]))
                polymer.update({"phi":phi_})
                polymer.update({"Q": Q_})

            # calculate the total energy
            w_minus = (w[0]-w[1])/2
            w_plus  = (w[0]+w[1])/2
            energy_total = self.cb.inner_product(w_minus,w_minus)/self.chi_n/self.cb.get_volume()
            energy_total -= self.cb.integral(w_plus)/self.cb.get_volume()

            for polymer in self.distinct_polymers:
                energy_total -= polymer["volume_fraction"]/polymer["alpha"]*np.log(polymer["Q"]/self.cb.get_volume())
                
            # calculate pressure field for the new field calculation
            xi = 0.5*(w[0]+w[1]-self.chi_n)

            # calculate output fields
            w_out[0] = self.chi_n*phi["B"] + xi
            w_out[1] = self.chi_n*phi["A"] + xi
                
            # keep the level of field value
            self.cb.zero_mean(w_out[0])
            self.cb.zero_mean(w_out[1])

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            w_diff = w_out - w
            multi_dot = self.cb.inner_product(w_diff[0],w_diff[0]) + self.cb.inner_product(w_diff[1],w_diff[1])
            multi_dot /= self.cb.inner_product(w[0],w[0]) + self.cb.inner_product(w[1],w[1]) + 1.0
            error_level = np.sqrt(multi_dot)

            # print iteration # and error levels and check the mass conservation
            mass_error = self.cb.integral(phi["A"] + phi["B"])/self.cb.get_volume() - 1.0
            
            if (self.box_is_altering):
                # Calculate stress
                stress_array = np.zeros(self.cb.get_dim())
                for polymer in self.distinct_polymers:
                    stress_array += polymer["volume_fraction"]/polymer["alpha"]/polymer["Q"] * \
                        np.array(polymer["pseudo"].get_stress()[-self.cb.get_dim():])
                #print(stress_array)
                error_level += np.sqrt(np.sum(stress_array)**2)
                print("%8d %12.3E " %
                (scft_iter, mass_error), end=" [ ")
                for polymer in self.distinct_polymers:
                    print("%13.7E " % (polymer["Q"]), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
                print("[", ",".join(["%10.7f" % (x) for x in self.cb.get_lx()[-self.cb.get_dim():]]), "]")
            else:
                print("%8d %12.3E " % (scft_iter, mass_error), end=" [ ")
                for polymer in self.distinct_polymers:
                    print("%13.7E " % (polymer["Q"]), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # conditions to end the iteration
            if error_level < self.tolerance:
                break

            # calculate new fields using simple and Anderson mixing
            if (self.box_is_altering):
                dlx = stress_array
                am_new  = np.concatenate((np.reshape(w,      2*self.cb.get_n_grid()), self.cb.get_lx()[-self.cb.get_dim():]))
                am_out  = np.concatenate((np.reshape(w_out,  2*self.cb.get_n_grid()), self.cb.get_lx()[-self.cb.get_dim():] + dlx))
                am_diff = np.concatenate((np.reshape(w_diff, 2*self.cb.get_n_grid()), stress_array))
                self.am.calculate_new_fields(am_new, am_out, am_diff, old_error_level, error_level)

                # set box size
                w[0] = am_new[0:self.cb.get_n_grid()]
                w[1] = am_new[self.cb.get_n_grid():2*self.cb.get_n_grid()]

                # restricting |dLx| to be less than 1 % of Lx
                old_lx = np.array(self.cb.get_lx()[-self.cb.get_dim():])
                new_lx = np.array(am_new[-self.cb.get_dim():])
                new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.01, 0.01)
                new_lx = (1 + new_dlx)*old_lx
                self.cb.set_lx(new_lx)
                # update bond parameters using new lx
                for polymer in self.distinct_polymers:
                    polymer["pseudo"].update()
            else:
                self.am.calculate_new_fields(
                np.reshape(w,      2*self.cb.get_n_grid()),
                np.reshape(w_out,  2*self.cb.get_n_grid()),
                np.reshape(w_diff, 2*self.cb.get_n_grid()), old_error_level, error_level)

        self.phi = phi
        self.w = w
        #return phi, Q, energy_total

    def get_concentrations(self,):
        return self.phi["A"], self.phi["B"]
    
    def get_fields(self,):
        return self.w[0], self.w[1]