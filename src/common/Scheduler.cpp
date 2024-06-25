#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cassert>
#include <map>
#include <set>

#include "Scheduler.h"

Scheduler::Scheduler(std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators, const int N_STREAM)
{
    try
    {
        int min_stream, minimum_time;
        int job_finish_time[N_STREAM] = {0,};

        std::vector<std::string> job_queue[N_STREAM];
        auto propagator_hierarchies = make_propagator_hierarchies(computation_propagators);

        // For height of propagator
        for(size_t current_height=0; current_height<propagator_hierarchies.size(); current_height++)
        {
            auto& same_height_propagators = propagator_hierarchies[current_height];
            std::vector<std::tuple<std::string, int>> Key_resolved_time;
            // Determine when propagator is ready to be computed, i.e., find dependencies resolved time.
            for(size_t i=0; i<same_height_propagators.size(); i++)
            {
                const auto& key = same_height_propagators[i];
                int max_resolved_time = 0;
                for(size_t j=0; j<computation_propagators[key].deps.size(); j++)
                {
                    const auto& sub_key = std::get<0>(computation_propagators[key].deps[j]);
                    int sub_n_segment = std::max(std::get<1>(computation_propagators[key].deps[j]),1); // add 1, if it is 0
                    #ifndef NDEBUG
                    if (stream_start_finish.find(sub_key) == stream_start_finish.end())
                        throw_with_line_number("Could not find [" + sub_key + "] in stream_start_finish.");
                    #endif
                    int sub_resolved_time = std::get<1>(stream_start_finish[sub_key]) + sub_n_segment; 
                    if (max_resolved_time == 0 || max_resolved_time < sub_resolved_time)
                        max_resolved_time = sub_resolved_time;
                }
                resolved_time[key] = max_resolved_time;
                Key_resolved_time.push_back(std::make_tuple(key, max_resolved_time));
            }

            // Sort propagators with time that they are ready
            std::sort(Key_resolved_time.begin(), Key_resolved_time.end(),
                [](auto const &t1, auto const &t2) {return std::get<1>(t1) < std::get<1>(t2);}
            );

            // for(int i=0; i<Key_resolved_time.size(); i++)
            // {
            //     const auto& key = std::get<0>(Key_resolved_time[i]);
            //     std::cout << key << ":\n\t";
            //     std::cout << "max_n_segment: " << computation_propagators[key].max_n_segment;
            //     std::cout << ", max_resolved_time: " << resolved_time[key] << std::endl;
            // }

            // Add job to compute propagators 
            for(size_t i=0; i<Key_resolved_time.size(); i++)
            {
                // Find index of stream that has minimum job_finish_time
                min_stream = 0;
                minimum_time = job_finish_time[0];
                for(int j=1; j<N_STREAM; j++)
                {
                    if(job_finish_time[j] < minimum_time)
                    {
                        min_stream = j;
                        minimum_time = job_finish_time[j];
                    }
                }
                // Add job at stream[min_stream]
                const auto& key = std::get<0>(Key_resolved_time[i]);
                int max_n_segment = std::max(computation_propagators[key].max_n_segment, 1); // if max_n_segment is 0, add 1
                int job_start_time = std::max(job_finish_time[min_stream], resolved_time[key]);
                // std::cout << key << ", " << min_stream << ", " << job_start_time << ", " << job_start_time+max_n_segment << std::endl;
                stream_start_finish[key] = std::make_tuple(min_stream, job_start_time, job_start_time + max_n_segment);
                job_finish_time[min_stream] = job_start_time + max_n_segment;
                job_queue[min_stream].push_back(key);

                // for(int j=0; j<N_STREAM; j++)
                // {
                //     std::cout << "(" << j << ": " << job_finish_time[j] << "), ";
                // }
                // std::cout << std::endl;
            }

        }

        // Sort propagators with starting time
        for(const auto& item : stream_start_finish)
            sorted_propagator_with_start_time.push_back(std::make_tuple(item.first, std::get<1>(item.second)));
        std::sort(sorted_propagator_with_start_time.begin(), sorted_propagator_with_start_time.end(),
            [](auto const &t1, auto const &t2) {return std::get<1>(t1) < std::get<1>(t2);}
        );

        // Collect time stamp
        std::set<int, std::less<int>> time_stamp_set;
        for(size_t i=0; i<sorted_propagator_with_start_time.size(); i++)
        {
            auto& key = std::get<0>(sorted_propagator_with_start_time[i]);
            int start_time = std::get<1>(sorted_propagator_with_start_time[i]);
            int finish_time = start_time + std::max(computation_propagators[key].max_n_segment, 1); // if max_n_segment is 0, add 1
            // std::cout << key << ":\n\t";
            // std::cout << "max_n_segment: " << computation_propagators[key].max_n_segment;
            // std::cout << ", start_time: " << start_time;
            // std::cout << ", finish_time: " << finish_time << std::endl;
            time_stamp_set.insert(start_time);
            time_stamp_set.insert(finish_time);
        }
        std::copy(time_stamp_set.begin(), time_stamp_set.end(), std::back_inserter(time_stamp));

        // for(int i=0; i<time_stamp.size()-1; i++)
        // {
        //     std::cout << "time_stamp" << i << ", " << time_stamp[i] << std::endl;
        // }

        // // scheduling
        // for(int s=0; s<N_STREAM; s++)
        // {         
        //     for(int i=0; i<job_queue[s].size()-1; i++)
        //     {
        //        std::cout << job_queue[s][i] << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        // For each stream, make iterator
        std::vector<std::string>::iterator iters[N_STREAM];
        for(int s=0; s<N_STREAM; s++)
            iters[s] = job_queue[s].begin();

        // For each time stamp
        for(size_t i=0; i<time_stamp.size()-1; i++)
        {
            // std::cout << time_stamp[i]+1 << ", " << time_stamp[i+1] << std::endl;
            std::vector<std::tuple<std::string, int, int>> parallel_job;

            // For each stream
            for(int s=0; s<N_STREAM; s++)
            {
                // If iters[s] (propagator iter) is not the end of job_queue
                if(iters[s] != job_queue[s].end())
                {
                    // If the finishing time of current propagator is smaller than finishing time of current time span, move to the next propagator of iters[s]. 
                    if(time_stamp[i+1] > std::get<2>(stream_start_finish[*iters[s]]))
                        iters[s]++;
                } 

                // If iters[s] (propagator iter) is not the end of job_queue
                if(iters[s] != job_queue[s].end())
                {
                    // std::cout << "\t*iters[s] " << *iters[s] << std::endl;
                    // std::cout << "\ttime_stamp " << time_stamp[i] << ", " << time_stamp[i+1] << std::endl;
                    // std::cout << "\tstream_start_finish " << std::get<1>(stream_start_finish[*iters[s]]) << ", " << std::get<2>(stream_start_finish[*iters[s]]) << std::endl;

                    // If the time span is in between starting time to finishing time, add propagator job to the parallel_job
                    if( time_stamp[i] >= std::get<1>(stream_start_finish[*iters[s]]) 
                        && time_stamp[i+1] <= std::get<2>(stream_start_finish[*iters[s]]))
                    {
                        int n_segment_from, n_segment_to;

                        // If max_n_segment is 0, skip propagator iterations 
                        if(computation_propagators[*iters[s]].max_n_segment == 0)
                        {
                            n_segment_from = 1;
                            n_segment_to = 0;
                        }
                        // Set range of n_segment to be computed
                        else
                        {
                            n_segment_from = 1+time_stamp[i]-std::get<1>(stream_start_finish[*iters[s]]);
                            n_segment_to = time_stamp[i+1]-std::get<1>(stream_start_finish[*iters[s]]);
                        }

                        // std::cout << "\t\t" << *iters[s] << ": " << n_segment_from << ", " << n_segment_to << std::endl;
                        parallel_job.push_back(std::make_tuple(*iters[s], n_segment_from, n_segment_to));
                    }
                }
            }
            schedule.push_back(parallel_job);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<std::vector<std::string>> Scheduler::make_propagator_hierarchies(
    std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators)
{
    try
    {
        std::vector<std::vector<std::string>> propagator_hierarchies;
        int current_height = 0;
        std::vector<std::string> same_height_propagators; // key
        std::vector<std::string> remaining_branches;

        for(const auto& item : computation_propagators)
            remaining_branches.push_back(item.first);

        while(!remaining_branches.empty())
        {
            same_height_propagators.clear();
            for(size_t i=0; i<remaining_branches.size(); i++)
            {
                if (current_height == PropagatorCode::get_height_from_key(remaining_branches[i]))
                    same_height_propagators.push_back(remaining_branches[i]);
            }
            if (!same_height_propagators.empty())
            {
                for(size_t i=0; i<same_height_propagators.size(); i++)
                    remaining_branches.erase(std::remove(remaining_branches.begin(), remaining_branches.end(), same_height_propagators[i]), remaining_branches.end());
                propagator_hierarchies.push_back(same_height_propagators);
            }
            current_height++;
        }

        // for(int i=0; i<propagator_hierarchies.size(); i++)
        // {
        //     std::cout << "Height:" << i << std::endl;
        //     for(const auto &item: propagator_hierarchies[i])
        //         std::cout << item << ": " << PropagatorCode::get_height_from_key(item) << std::endl;
        // }

        // for(const auto& item: computation_propagators)
        // {
        //     auto& key = item.first;

        //     int max_n_segment = item.second.max_n_segment;
        //     //int monomer_type = item.second.monomer_type;
        //     int height = item.second.height;
        //     auto& deps = item.second.deps;

        //     if (current_height < height)
        //     {
        //         propagator_hierarchies.push_back(same_height_propagators);
        //         same_height_propagators.clear();
        //         current_height = height;
        //     }
        //     same_height_propagators.push_back(key);
        // }

        // propagator_hierarchies.push_back(same_height_propagators);
        // same_height_propagators.clear();
        return propagator_hierarchies;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<std::vector<std::tuple<std::string, int, int>>>& Scheduler::get_schedule()
{
    return schedule;
}
void Scheduler::display(std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators)
{
    for(size_t i=0; i<sorted_propagator_with_start_time.size(); i++)
    {
        auto& key = std::get<0>(sorted_propagator_with_start_time[i]);
        int start_time = std::get<1>(sorted_propagator_with_start_time[i]);
        int finish_time = start_time + computation_propagators[key].max_n_segment;
        std::cout << key << ":\n\t";
        std::cout << "max_n_segment: " << computation_propagators[key].max_n_segment;
        std::cout << ", start_time: " << start_time;
        std::cout << ", finish_time: " << finish_time << std::endl;
    }

    for(size_t i=0; i<schedule.size(); i++)
    {
        std::cout << "time: " << time_stamp[i]+1 << "-" << time_stamp[i+1] << std::endl;
        auto& parallel_job = schedule[i];
        for(size_t j=0; j<parallel_job.size(); j++)
            std::cout << "\t" << std::get<0>(parallel_job[j]) << ": " <<  std::get<1>(parallel_job[j]) << ", " <<  std::get<2>(parallel_job[j]) << std::endl;
    }
}