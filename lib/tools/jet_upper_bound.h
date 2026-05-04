/******************************************************************************
 * jet_upper_bound.h
 *
 * Optional JET preprocessing for VieCut.
 *****************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "common/configuration.h"
#include "common/definitions.h"
#include "tools/timer.h"

#ifdef VIECUT_ENABLE_JET
#include "Kokkos_Core.hpp"
#include "jet.h"
#include "jet_config.h"
#include "jet_defs.h"
#endif

namespace jet_upper_bound {

struct Result {
    bool success = false;
    EdgeWeight cut = 0;
    double time = 0.0;
    double conversion_time = 0.0;
    double partition_time = 0.0;
    std::string message;
    std::vector<unsigned char> partition;
};

#ifdef VIECUT_ENABLE_JET

class KokkosRuntime {
 public:
    KokkosRuntime() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            owns_runtime_ = true;
        }
    }

    ~KokkosRuntime() {
        if (owns_runtime_ && Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }

 private:
    bool owns_runtime_ = false;
};

inline bool loadConfigFile(const std::string& path,
                           jet_partitioner::config_t* config) {
    if (path.empty()) {
        return true;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }

    std::string lines[6];
    int reads = 0;
    for (int i = 0; i < 6; ++i) {
        if (in >> lines[i]) {
            ++reads;
        }
    }

    if (reads < 4) {
        return false;
    }

    config->coarsening_alg = std::stoi(lines[0]);
    config->num_parts = std::stoi(lines[1]);
    config->num_iter = std::stoi(lines[2]);
    config->max_imb_ratio = std::stod(lines[3]);
    if (reads >= 5) {
        config->ultra_settings = std::stoi(lines[4]);
    }
    if (reads >= 6) {
        config->min_cut_mode = std::stoi(lines[5]);
    }

    return true;
}

template <typename GraphPtr>
Result compute(GraphPtr G) {
    Result result;
    timer total_timer;
    auto cfg = configuration::getConfig();

    if (!G) {
        result.message = "empty graph";
        return result;
    }

    const NodeID n = G->number_of_nodes();
    const EdgeID directed_edges = G->number_of_edges();
    if (n == 0 || directed_edges == 0) {
        result.success = true;
        result.cut = 0;
        result.time = total_timer.elapsed();
        return result;
    }

    if (n > static_cast<uint64_t>(
            std::numeric_limits<jet_partitioner::ordinal_t>::max())) {
        result.message = "graph has too many nodes for JET ordinal_t";
        result.time = total_timer.elapsed();
        return result;
    }

    if (directed_edges > static_cast<uint64_t>(
            std::numeric_limits<jet_partitioner::big_offset_t>::max())) {
        result.message = "graph has too many directed edges for JET offsets";
        result.time = total_timer.elapsed();
        return result;
    }

    jet_partitioner::config_t jet_cfg;
    if (!loadConfigFile(cfg->jet_config_file, &jet_cfg)) {
        result.message = "could not load JET config file";
        result.time = total_timer.elapsed();
        return result;
    }
    jet_cfg.num_parts = 2;
    jet_cfg.num_iter = std::max<size_t>(1, cfg->jet_num_iterations);
    jet_cfg.min_cut_mode = true;
    jet_cfg.verbose = false;

    using namespace jet_partitioner;

    KokkosRuntime runtime;

    timer conversion_timer;
    Kokkos::View<big_offset_t*, Kokkos::HostSpace> row_map_h(
        Kokkos::ViewAllocateWithoutInitializing("viecut row map host"),
        static_cast<size_t>(n) + 1);
    Kokkos::View<ordinal_t*, Kokkos::HostSpace> entries_h(
        Kokkos::ViewAllocateWithoutInitializing("viecut entries host"),
        static_cast<size_t>(directed_edges));
    Kokkos::View<big_val_t*, Kokkos::HostSpace> values_h(
        Kokkos::ViewAllocateWithoutInitializing("viecut values host"),
        static_cast<size_t>(directed_edges));

    big_offset_t edge_pos = 0;
    row_map_h(0) = 0;
    for (NodeID u : G->nodes()) {
        for (EdgeID e : G->edges_of(u)) {
            entries_h(edge_pos) =
                static_cast<ordinal_t>(G->getEdgeTarget(u, e));
            values_h(edge_pos) =
                static_cast<big_val_t>(G->getEdgeWeight(u, e));
            ++edge_pos;
        }
        row_map_h(static_cast<size_t>(u) + 1) = edge_pos;
    }

    Kokkos::View<big_offset_t*, Device> row_map(
        Kokkos::ViewAllocateWithoutInitializing("viecut row map"),
        static_cast<size_t>(n) + 1);
    Kokkos::View<ordinal_t*, Device> entries(
        Kokkos::ViewAllocateWithoutInitializing("viecut entries"),
        static_cast<size_t>(directed_edges));
    big_wgt_vt values(
        Kokkos::ViewAllocateWithoutInitializing("viecut values"),
        static_cast<size_t>(directed_edges));

    Kokkos::deep_copy(row_map, row_map_h);
    Kokkos::deep_copy(entries, entries_h);
    Kokkos::deep_copy(values, values_h);

    typename big_matrix_t::staticcrsgraph_type graph(entries, row_map);
    big_matrix_t jet_graph("viecut graph", n, values, graph);
    big_wgt_vt vertex_weights("viecut vertex weights", n);
    Kokkos::deep_copy(vertex_weights, 1);
    result.conversion_time = conversion_timer.elapsed();

    big_val_t best_cut = std::numeric_limits<big_val_t>::max();
    std::vector<unsigned char> best_partition;
    const int iterations = jet_cfg.num_iter;
    jet_cfg.num_iter = 1;
    timer partition_timer;
    for (int i = 0; i < iterations; ++i) {
        big_val_t cut = 0;
        experiment_data<big_val_t> experiment;
        part_vt partition = partition_big(
            cut, jet_cfg, jet_graph, vertex_weights, false, experiment);
        if (cut < best_cut) {
            best_cut = cut;
            if (cfg->save_cut) {
                part_mt partition_h = Kokkos::create_mirror_view(partition);
                Kokkos::deep_copy(partition_h, partition);
                best_partition.resize(n);
                for (NodeID u = 0; u < n; ++u) {
                    best_partition[u] =
                        static_cast<unsigned char>(partition_h(u) != 0);
                }
            }
        }
    }
    result.partition_time = partition_timer.elapsed();

    result.success = true;
    result.cut = static_cast<EdgeWeight>(best_cut);
    result.time = total_timer.elapsed();
    result.partition = std::move(best_partition);
    return result;
}

#else

class KokkosRuntime {
 public:
    KokkosRuntime() = default;
};

template <typename GraphPtr>
Result compute(GraphPtr) {
    Result result;
    result.message = "VieCut was built without VIECUT_ENABLE_JET";
    return result;
}

#endif

}  // namespace jet_upper_bound
