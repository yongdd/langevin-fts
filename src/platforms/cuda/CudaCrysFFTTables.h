/**
 * @file CudaCrysFFTTables.h
 * @brief k-space tables shared by the CrysFFT objects of the streams of one solver.
 *
 * A solver keeps one CrysFFT object per stream (see CudaCrysFFTBase). Their
 * tables (Boltzmann factors, stress multipliers) depend only on the grid, the
 * cell and a key, so the objects share one CudaCrysFFTTables:
 *
 * - The cell lives here. set_cell() by any object changes it for all of them:
 *   it waits for the device (kernels of every stream may still read the
 *   tables), drops the tables, keeps their buffers for the tables built next
 *   and advances the generation.
 * - A missing table is built under the mutex, for the cell held here, by the
 *   object that needs it first.
 * - Each object looks tables up through its CudaCrysFFTTableView, which caches
 *   the tables it has seen without locking and discards them when the
 *   generation changes, so no object keeps a table of an older cell.
 *
 * set_cell() must not run while any stream computes.
 */

#ifndef CUDA_CRYS_FFT_TABLES_H_
#define CUDA_CRYS_FFT_TABLES_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "CudaCommon.h"

/// Number of CrysFFT tables built so far in this process (tests check that the
/// propagator and stress loops build none).
inline std::atomic<std::int64_t>& cuda_crysfft_table_build_counter()
{
    static std::atomic<std::int64_t> count{0};
    return count;
}

/// Key of a CrysFFT table: a class-specific kind, an axis mask or multiplier
/// type, and the coefficient (b²·ds/divisor, or 0 when the table has none).
struct CudaCrysFFTTableKey
{
    int kind;
    int index;
    double coeff;

    bool operator<(const CudaCrysFFTTableKey& other) const
    {
        return std::tie(kind, index, coeff) < std::tie(other.kind, other.index, other.coeff);
    }
};

/**
 * @brief Tables of one CrysFFT class shared by the objects of the streams.
 *
 * @tparam Table Move-only device table; all tables of a class have one shape,
 *               so the buffers of dropped tables can hold any new table.
 */
template <typename Table>
class CudaCrysFFTTables
{
public:
    explicit CudaCrysFFTTables(const std::array<double, 6>& cell) : cell_(cell) {}

    /// Generation of the current tables; it changes with every cell change.
    std::uint64_t generation() const { return generation_.load(std::memory_order_acquire); }

    /// Change the cell of the tables (no-op for the current cell).
    void set_cell(const std::array<double, 6>& cell)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cell == cell_)
            return;
        // Kernels of any stream may still read the tables whose buffers the next builds rewrite
        CUDA_CHECK(cudaDeviceSynchronize());
        for (auto& kv : tables_)
            spare_.push_back(std::move(kv.second));
        tables_.clear();
        cell_ = cell;
        generation_.fetch_add(1, std::memory_order_acq_rel);
    }

    /**
     * @brief The table of `key`, built when missing.
     *
     * build(cell, buffers) makes the table for `cell`, filling `buffers`, which
     * hold a dropped table or are empty (allocate them then). It runs with the
     * mutex held and must complete its uploads before it returns.
     */
    template <typename Build>
    const Table& get(const CudaCrysFFTTableKey& key, Build&& build)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tables_.find(key);
        if (it == tables_.end())
        {
            Table buffers{};
            if (!spare_.empty())
            {
                buffers = std::move(spare_.back());
                spare_.pop_back();
            }
            it = tables_.emplace(key, build(cell_, std::move(buffers))).first;
            cuda_crysfft_table_build_counter().fetch_add(1, std::memory_order_relaxed);
        }
        return it->second;
    }

private:
    std::mutex mutex_;
    std::array<double, 6> cell_;                     ///< Cell of the tables
    std::atomic<std::uint64_t> generation_{0};
    std::map<CudaCrysFFTTableKey, Table> tables_;    ///< Node-based: references stay valid until the next cell change
    std::vector<Table> spare_;                       ///< Buffers of the tables dropped by the last cell change
};

/**
 * @brief One object's view of the shared tables.
 *
 * Caches the tables the object has looked up, without locking, until the
 * generation of the shared tables changes.
 */
template <typename Table>
class CudaCrysFFTTableView
{
public:
    explicit CudaCrysFFTTableView(std::shared_ptr<CudaCrysFFTTables<Table>> tables)
        : tables_(std::move(tables)), generation_(tables_->generation()) {}

    /// The shared tables (for clone_sharing_tables() and set_cell()).
    const std::shared_ptr<CudaCrysFFTTables<Table>>& shared() const { return tables_; }

    /// Discard the cached tables if the cell changed; returns true if it did.
    bool refresh()
    {
        const std::uint64_t generation = tables_->generation();
        if (generation == generation_)
            return false;
        seen_.clear();
        generation_ = generation;
        return true;
    }

    /// The table of `key` (see CudaCrysFFTTables::get()).
    template <typename Build>
    const Table& get(const CudaCrysFFTTableKey& key, Build&& build)
    {
        refresh();
        auto it = seen_.find(key);
        if (it == seen_.end())
            it = seen_.emplace(key, &tables_->get(key, std::forward<Build>(build))).first;
        return *it->second;
    }

    /// Drop the cache and the reference to the shared tables (release paths; does not throw).
    void reset() noexcept
    {
        seen_.clear();
        tables_.reset();
    }

private:
    std::shared_ptr<CudaCrysFFTTables<Table>> tables_;
    std::uint64_t generation_;
    std::map<CudaCrysFFTTableKey, const Table*> seen_;
};

#endif  // CUDA_CRYS_FFT_TABLES_H_
