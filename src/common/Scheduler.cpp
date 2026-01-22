/**
 * @file Scheduler.cpp
 * @brief Implementation of parallel propagator scheduling.
 *
 * Creates an execution schedule for computing propagators across multiple
 * GPU streams or CPU threads. Respects dependency ordering while maximizing
 * parallelism through list scheduling with greedy stream assignment.
 *
 * **Scheduling Algorithm:**
 *
 * 1. **Group by height**: Propagators grouped by dependency depth
 * 2. **Resolve dependencies**: Find when each propagator can start
 * 3. **Greedy assignment**: Assign to stream with earliest available time
 * 4. **Time slicing**: Divide into intervals for interleaved progress
 *
 * **Example:**
 *
 * Time 0-10:  [("A+0", 0, 10), ("B+0", 0, 10)]  // A and B in parallel
 * Time 10-15: [("A+0", 10, 15)]                  // Only A continues
 * Time 15-20: [("(AB)C+0", 0, 5)]               // Dependent C starts
 *
 * @see PropagatorComputationOptimizer for propagator dependency analysis
 */

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "Scheduler.h"

/**
 * @brief Find stream with earliest available time.
 */
int Scheduler::find_earliest_stream(const std::vector<int>& stream_available_time)
{
    return std::min_element(stream_available_time.begin(), stream_available_time.end())
           - stream_available_time.begin();
}

/**
 * @brief Group propagators by dependency height.
 *
 * Height 0 = no dependencies, Height 1 = depends only on height 0, etc.
 */
std::vector<std::vector<std::string>> Scheduler::group_by_height(
    const std::map<std::string, ComputationEdge, ComparePropagatorKey>& propagators)
{
    // Find maximum height
    int max_height = 0;
    for (const auto& [key, _] : propagators)
    {
        int height = PropagatorCode::get_height_from_key(key);
        max_height = std::max(max_height, height);
    }

    // Group by height
    std::vector<std::vector<std::string>> groups(max_height + 1);
    for (const auto& [key, _] : propagators)
    {
        int height = PropagatorCode::get_height_from_key(key);
        groups[height].push_back(key);
    }

    return groups;
}

/**
 * @brief Construct scheduler and compute the execution schedule.
 */
Scheduler::Scheduler(std::map<std::string, ComputationEdge, ComparePropagatorKey> propagators, int n_streams)
{
    try
    {
        std::vector<int> stream_available_time(n_streams, 0);
        std::vector<std::vector<std::string>> stream_jobs(n_streams);

        auto height_groups = group_by_height(propagators);

        // Process propagators level by level (lower height first)
        for (const auto& same_height_keys : height_groups)
        {
            // Compute dependency resolved time for each propagator
            std::vector<std::pair<std::string, int>> ready_propagators;
            for (const auto& key : same_height_keys)
            {
                int resolved_time = 0;
                for (const auto& [dep_key, dep_n_segment, dep_n_repeated] : propagators[key].deps)
                {
                    #ifndef NDEBUG
                    if (job_assignment.find(dep_key) == job_assignment.end())
                        throw_with_line_number("Could not find dependency [" + dep_key + "] in job_assignment.");
                    #endif
                    int dep_start = std::get<1>(job_assignment[dep_key]);
                    int dep_duration = std::max(dep_n_segment, 1);
                    resolved_time = std::max(resolved_time, dep_start + dep_duration);
                }
                dependency_resolved_time[key] = resolved_time;
                ready_propagators.emplace_back(key, resolved_time);
            }

            // Sort by resolved time (earlier ready = scheduled first)
            std::sort(ready_propagators.begin(), ready_propagators.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            // Greedily assign each propagator to earliest available stream
            for (const auto& [key, resolved_time] : ready_propagators)
            {
                int stream = find_earliest_stream(stream_available_time);
                int start_time = std::max(stream_available_time[stream], resolved_time);
                int duration = std::max(propagators[key].max_n_segment, 1);
                int end_time = start_time + duration;

                job_assignment[key] = {stream, start_time, end_time};
                stream_available_time[stream] = end_time;
                stream_jobs[stream].push_back(key);
            }
        }

        // Build sorted_jobs list
        for (const auto& [key, assignment] : job_assignment)
            sorted_jobs.emplace_back(key, std::get<1>(assignment));
        std::sort(sorted_jobs.begin(), sorted_jobs.end(),
            [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });

        // Collect unique time points (when jobs start or end)
        std::set<int> time_point_set;
        for (const auto& [key, start_time] : sorted_jobs)
        {
            int duration = std::max(propagators[key].max_n_segment, 1);
            time_point_set.insert(start_time);
            time_point_set.insert(start_time + duration);
        }
        time_points.assign(time_point_set.begin(), time_point_set.end());

        // Build schedule: for each time interval, list active jobs
        std::vector<std::vector<std::string>::iterator> stream_iters(n_streams);
        for (int s = 0; s < n_streams; s++)
            stream_iters[s] = stream_jobs[s].begin();

        for (size_t t = 0; t + 1 < time_points.size(); t++)
        {
            int interval_start = time_points[t];
            int interval_end = time_points[t + 1];
            std::vector<std::tuple<std::string, int, int>> active_jobs;

            for (int s = 0; s < n_streams; s++)
            {
                // Advance iterator if current job has finished
                if (stream_iters[s] != stream_jobs[s].end())
                {
                    int job_end = std::get<2>(job_assignment[*stream_iters[s]]);
                    if (interval_end > job_end)
                        stream_iters[s]++;
                }

                // Check if current job is active in this interval
                if (stream_iters[s] != stream_jobs[s].end())
                {
                    const std::string& key = *stream_iters[s];
                    int job_start = std::get<1>(job_assignment[key]);
                    int job_end = std::get<2>(job_assignment[key]);

                    if (interval_start >= job_start && interval_end <= job_end)
                    {
                        int segment_from, segment_to;
                        if (propagators[key].max_n_segment == 0)
                        {
                            segment_from = 0;
                            segment_to = 0;
                        }
                        else
                        {
                            segment_from = interval_start - job_start;
                            segment_to = interval_end - job_start;
                        }
                        active_jobs.emplace_back(key, segment_from, segment_to);
                    }
                }
            }
            schedule.push_back(active_jobs);
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::vector<std::vector<std::tuple<std::string, int, int>>>& Scheduler::get_schedule()
{
    return schedule;
}

void Scheduler::display(std::map<std::string, ComputationEdge, ComparePropagatorKey> propagators)
{
    std::cout << "=== Propagator Schedule ===" << std::endl;

    for (const auto& [key, start_time] : sorted_jobs)
    {
        int duration = propagators[key].max_n_segment;
        std::cout << key << ":" << std::endl;
        std::cout << "    n_segment=" << duration
                  << ", start=" << start_time
                  << ", end=" << start_time + duration << std::endl;
    }

    std::cout << "\n=== Time Slices ===" << std::endl;
    for (size_t t = 0; t < schedule.size(); t++)
    {
        std::cout << "Time " << time_points[t] << "-" << time_points[t + 1] << ":" << std::endl;
        for (const auto& [key, from, to] : schedule[t])
            std::cout << "    " << key << ": segments " << from << "-" << to << std::endl;
    }
}
