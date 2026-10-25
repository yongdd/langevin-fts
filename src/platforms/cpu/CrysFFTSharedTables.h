/**
 * @file CrysFFTSharedTables.h
 * @brief Read-only k-space tables of a CPU CrysFFT engine, shared by its calling threads.
 *
 * The propagator loop calls one engine from several OpenMP threads, each with
 * its own contour step. The tables (Boltzmann factors, stress multipliers)
 * depend only on the key and the cell, so each one is built once, under a
 * mutex, and read by every thread. A thread remembers, per cache, the tables it
 * has looked up and the one it selected last, so the usual lookup takes no
 * lock. invalidate() (a cell change, made while no thread uses the engine)
 * drops the tables and advances the generation that makes every thread forget
 * its pointers.
 */

#ifndef CRYS_FFT_SHARED_TABLES_H_
#define CRYS_FFT_SHARED_TABLES_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>

template <typename Key, typename Table>
class SharedTableCache
{
public:
    SharedTableCache() = default;
    SharedTableCache(const SharedTableCache&) = delete;
    SharedTableCache& operator=(const SharedTableCache&) = delete;

    /// Table of key (built by build() on first use by any thread), made the
    /// calling thread's current table.
    template <typename Build>
    const Table& select(const Key& key, Build&& build) const
    {
        View& view = thread_view();
        if (view.current == nullptr || !(view.current_key == key))
        {
            view.current = &lookup(view, key, build);
            view.current_key = key;
        }
        return *view.current;
    }

    /// The calling thread's current table: nullptr when it has selected none
    /// since the last invalidate().
    const Table* current() const { return thread_view().current; }

    /// Table of key (built by build() on first use by any thread), without selecting it.
    template <typename Build>
    const Table& get(const Key& key, Build&& build) const
    {
        return lookup(thread_view(), key, build);
    }

    /// Drop every table (cell change); no thread may use the cache meanwhile.
    void invalidate()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tables_.clear();
        generation_.fetch_add(1, std::memory_order_acq_rel);
    }

private:
    /// What one thread has looked up (pointers into tables_)
    struct View
    {
        std::map<Key, const Table*> seen;
        const Table* current = nullptr;
        Key current_key{};
        uint64_t generation = 0;
    };

    template <typename Build>
    const Table& lookup(View& view, const Key& key, Build& build) const
    {
        auto seen = view.seen.find(key);
        if (seen != view.seen.end())
            return *seen->second;
        const Table* table;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = tables_.find(key);
            if (it == tables_.end())
                it = tables_.emplace(key, build()).first;   // map nodes keep their address
            table = &it->second;
        }
        view.seen.emplace(key, table);
        return *table;
    }

    View& thread_view() const
    {
        // Each thread remembers the view it used last, tagged with the cache's
        // id, so the usual case (one engine per thread) takes no lock; ids are
        // not reused, so a remembered pointer of a destroyed cache is never followed
        struct LastSlot { uint64_t instance_id = 0; View* view = nullptr; };
        thread_local LastSlot last;
        if (last.instance_id != instance_id_)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last.view = &views_[std::this_thread::get_id()];
            last.instance_id = instance_id_;
        }
        View* view = last.view;
        const uint64_t generation = generation_.load(std::memory_order_acquire);
        if (view->generation != generation)
        {
            view->seen.clear();
            view->current = nullptr;
            view->generation = generation;
        }
        return *view;
    }

    mutable std::mutex mutex_;
    mutable std::map<Key, Table> tables_;
    mutable std::unordered_map<std::thread::id, View> views_;   ///< owned here, released with the cache
    std::atomic<uint64_t> generation_{1};
    inline static std::atomic<uint64_t> next_instance_id_{1};
    const uint64_t instance_id_{next_instance_id_.fetch_add(1, std::memory_order_relaxed)};
};

#endif  // CRYS_FFT_SHARED_TABLES_H_
