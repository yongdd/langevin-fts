#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cassert>
#include <map>
#include <set>

#include "Scheduler.h"

Scheduler::Scheduler(std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches, const int N_STREAM)
{
    try
    {
        int min_stream, minimum_time;
        int job_finish_time[N_STREAM] = {0,};

        std::vector<std::string> job_queue[N_STREAM];
        auto branch_hierarchies = make_branch_hierarchies(unique_branches);
        for(int current_height=0; current_height<branch_hierarchies.size(); current_height++)
        {
            auto& same_height_branches = branch_hierarchies[current_height];
            std::vector<std::tuple<std::string, int>> Key_resolved_time;
            // find dependencies resolved time
            for(int i=0; i<same_height_branches.size(); i++)
            {
                const auto& key = same_height_branches[i];
                int max_resolved_time = 0;
                for(int j=0; j<unique_branches[key].deps.size(); j++)
                {
                    const auto& sub_key = unique_branches[key].deps[j].first;
                    const auto& sub_n_segment = unique_branches[key].deps[j].second;
                    assert(stream_start_finish.count(sub_key) == 0 && "Could not find [" + sub_key + "] in stream_start_finish.");
                    int sub_resolved_time = std::get<1>(stream_start_finish[sub_key]) + sub_n_segment;
                    if (max_resolved_time == 0 || max_resolved_time < sub_resolved_time)
                        max_resolved_time = sub_resolved_time;
                }
                resolved_time[key] = max_resolved_time;
                Key_resolved_time.push_back(std::make_tuple(key, max_resolved_time));
            }

            // sort branches on the basis of resolved time
            std::sort(Key_resolved_time.begin(), Key_resolved_time.end(),
                [](auto const &t1, auto const &t2) {return std::get<1>(t1) < std::get<1>(t2);}
            );

            // for(int i=0; i<Key_resolved_time.size(); i++)
            // {
            //     const auto& key = std::get<0>(Key_resolved_time[i]);
            //     std::cout << key << ":\n\t";
            //     std::cout << "max_n_segment: " << unique_branches[key].max_n_segment;
            //     std::cout << ", max_resolved_time: " << resolved_time[key] << std::endl;
            // }

            // add jobs 
            for(int i=0; i<Key_resolved_time.size(); i++)
            {
                // find index of stream that has minimum job_finish_time
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
                // add job at stream[min_stream]
                const auto& key = std::get<0>(Key_resolved_time[i]);
                int max_n_segment = unique_branches[key].max_n_segment;
                int job_start_time = std::max(job_finish_time[min_stream], resolved_time[key]);
                // std::cout << key << ", " << min_stream << ", " << job_start_time << ", " << job_start_time+max_n_segment << std::endl;
                stream_start_finish[key] = std::make_tuple(min_stream, job_start_time, job_start_time+max_n_segment);
                job_finish_time[min_stream] = job_start_time + max_n_segment;
                job_queue[min_stream].push_back(key);

                // for(int j=0; j<N_STREAM; j++)
                // {
                //     std::cout << "(" << j << ": " << job_finish_time[j] << "), ";
                // }
                // std::cout << std::endl;
            }

        }

        // sort branches starting time
        for(const auto& item : stream_start_finish)
            sorted_branch_start_time.push_back(std::make_tuple(item.first, std::get<1>(item.second)));
        std::sort(sorted_branch_start_time.begin(), sorted_branch_start_time.end(),
            [](auto const &t1, auto const &t2) {return std::get<1>(t1) < std::get<1>(t2);}
        );

        // collect time stamp
        std::set<int, std::less<int>> time_stamp_set;
        for(int i=0; i<sorted_branch_start_time.size(); i++)
        {
            auto& key = std::get<0>(sorted_branch_start_time[i]);
            int start_time = std::get<1>(sorted_branch_start_time[i]);
            int finish_time = start_time + unique_branches[key].max_n_segment;
            // std::cout << key << ":\n\t";
            // std::cout << "max_n_segment: " << unique_branches[key].max_n_segment;
            // std::cout << ", start_time: " << start_time;
            // std::cout << ", finish_time: " << finish_time << std::endl;
            time_stamp_set.insert(start_time);
            time_stamp_set.insert(finish_time);
        }
        std::copy(time_stamp_set.begin(), time_stamp_set.end(), std::back_inserter(time_stamp));

        // scheduling
        // for(int s=0; s<N_STREAM; s++)
        // {         
        //     for(int i=0; i<job_queue[s].size()-1; i++)
        //     {
        //        std::cout << job_queue[s][i] << ", ";
        //     }
        //     std::cout << std::endl;
        // }

        std::vector<std::string>::iterator iters[N_STREAM];
        for(int s=0; s<N_STREAM; s++)
            iters[s] = job_queue[s].begin();


        for(int i=0; i<time_stamp.size()-1; i++)
        {
            //std::cout << time_stamp[i]+1 << ", " << time_stamp[i+1] << std::endl;
            std::vector<std::tuple<std::string, int, int>> parallel_job;
            for(int s=0; s<N_STREAM; s++)
            {
                // start_time < time_stamp[i]
                if(iters[s] != job_queue[s].end())
                {
                    if(time_stamp[i+1] > std::get<2>(stream_start_finish[*iters[s]]))
                        iters[s]++;
                }
                if(iters[s] != job_queue[s].end())
                {
                    if( time_stamp[i] >= std::get<1>(stream_start_finish[*iters[s]]) 
                        && time_stamp[i+1] <= std::get<2>(stream_start_finish[*iters[s]]))
                        {
                        int n_segment_from = 1+time_stamp[i]-std::get<1>(stream_start_finish[*iters[s]]);
                        int n_segment_to = time_stamp[i+1]-std::get<1>(stream_start_finish[*iters[s]]);

                        //std::cout << *iters[s] << ": " << n_segment_from << ", " << n_segment_to << std::endl;
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
std::vector<std::vector<std::string>> Scheduler::make_branch_hierarchies(
    std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches)
{
    try
    {
        std::vector<std::vector<std::string>> branch_hierarchies;
        int current_height = 0;
        std::vector<std::string> same_height_branches; // key
        for(const auto& item: unique_branches)
        {
            auto& key = item.first;

            int max_n_segment = item.second.max_n_segment;
            //int monomer_type = item.second.monomer_type;
            int height = item.second.height;
            auto& deps = item.second.deps;

            if (current_height < height)
            {
                branch_hierarchies.push_back(same_height_branches);
                same_height_branches.clear();
                current_height = height;
            }
            same_height_branches.push_back(key);
        }
        branch_hierarchies.push_back(same_height_branches);
        same_height_branches.clear();
        return branch_hierarchies;
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
void Scheduler::display(std::map<std::string, UniqueEdge, std::greater<std::string>> unique_branches)
{
    for(int i=0; i<sorted_branch_start_time.size(); i++)
    {
        auto& key = std::get<0>(sorted_branch_start_time[i]);
        int start_time = std::get<1>(sorted_branch_start_time[i]);
        int finish_time = start_time + unique_branches[key].max_n_segment;
        std::cout << key << ":\n\t";
        std::cout << "max_n_segment: " << unique_branches[key].max_n_segment;
        std::cout << ", start_time: " << start_time;
        std::cout << ", finish_time: " << finish_time << std::endl;
    }

    for(int i=0; i<schedule.size(); i++)
    {
        std::cout << "time: " << time_stamp[i]+1 << "-" << time_stamp[i+1] << std::endl;
        auto& parallel_job = schedule[i];
        for(int j=0; j<parallel_job.size(); j++)
            std::cout << "\t" << std::get<0>(parallel_job[j]) << ": " <<  std::get<1>(parallel_job[j]) << ", " <<  std::get<2>(parallel_job[j]) << std::endl;
    }
}

