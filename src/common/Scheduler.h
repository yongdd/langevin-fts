/*----------------------------------------------------------
* This class schedules propagator calculations for parallel computation
*-----------------------------------------------------------*/

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"

class Scheduler
{
private:

    // Variables
    std::map<std::string, std::tuple<int, int, int>, ComparePropagatorKey> stream_start_finish; //stream_number, starting time, finishing time
    std::map<std::string, int> resolved_time; // when dependencies are resolved, e.g., when propagator is ready to be computed
    std::vector<std::tuple<std::string, int>> sorted_propagator_with_start_time;  // computation starting time for each propagator
    std::vector<int> time_stamp; // times that new jobs are joined or jobs are finished.
    std::vector<std::vector<std::tuple<std::string, int, int>>> schedule;   // job schedule for each time interval

    // Methods
    std::vector<std::vector<std::string>> make_propagator_hierarchies(
        std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators);
public:

    Scheduler(std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators, const int N_STREAM);
    ~Scheduler() {};
    std::vector<std::vector<std::tuple<std::string, int, int>>>& get_schedule();
    void display(std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators);
};
#endif
