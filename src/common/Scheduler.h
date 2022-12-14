/*----------------------------------------------------------
* This class schedules partial partition function calculations for parallel computation
*-----------------------------------------------------------*/

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Mixture.h"

class Scheduler
{
private:

    // variables
    std::map<std::string, std::tuple<int, int, int>, std::greater<std::string>> stream_start_finish; //stream_number, starting time, finishing time
    std::map<std::string, int> resolved_time; // when dependencies are resolved
    std::vector<std::tuple<std::string, int>> sorted_branch_start_time;  // start time for each branch
    std::vector<int> time_stamp; // times that new jobs are joined or jobs are finished.
    std::vector<std::vector<std::tuple<std::string, int, int>>> schedule;   // job schedule for each time interval

    // methods
    std::vector<std::vector<std::string>> make_branch_hierarchies(
        std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches);
public:

    Scheduler(std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches, const int N_STREAM);
    ~Scheduler() {};
    std::vector<std::vector<std::tuple<std::string, int, int>>>& get_schedule();
    void display(std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches);
};
#endif
