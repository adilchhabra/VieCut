/******************************************************************************
 * graph_class_guesser.cpp
 *
 * Standalone graph feature scan + class/preset guess utility.
 *
 *****************************************************************************/

#include <memory>
#include <string>

#include "data_structure/mutable_graph.h"
#include "io/graph_io.h"
#include "tlx/cmdline_parser.hpp"
#include "tlx/logger.hpp"
#include "tools/graph_features.h"
#include "tools/preset_selector.h"
#include "tools/timer.h"

typedef mutable_graph graph_type;
typedef std::shared_ptr<graph_type> GraphPtr;

int main(int argn, char** argv) {
    std::string graph_filename;
    bool verbose = false;

    tlx::CmdlineParser cmdl;
    cmdl.add_param_string("graph", graph_filename, "path to graph file");
    cmdl.add_flag('v', "verbose", verbose, "print additional details");

    if (!cmdl.process(argn, argv)) {
        return -1;
    }

    timer t;
    GraphPtr G = graph_io::readGraphWeighted<graph_type>(graph_filename);
    double io_time = t.elapsed();

    t.restart();
    autotune::GraphFeatures f = autotune::computeGraphFeatures(G);
    auto decision = autotune::recommendPreset(f);
    double feature_time = t.elapsed();

    std::cout << "FEATURES"
              << " graph=" << graph_filename
              << " n=" << f.nodes
              << " m=" << f.edges_undirected
              << " avg_degree=" << f.avg_degree
              << " density=" << f.density
              << " min_degree=" << f.min_degree
              << " degree_p50=" << f.degree_p50
              << " degree_p90=" << f.degree_p90
              << " degree_p99=" << f.degree_p99
              << " degree_ratio_p99_p50=" << f.degree_ratio_p99_p50
              << " degree_ratio_max_p90=" << f.degree_ratio_max_p90
              << " degree_tail_hill_alpha=" << f.degree_tail_hill_alpha
              << " degree_stddev=" << f.degree_stddev
              << " degree_cv=" << f.degree_cv
              << " max_degree=" << f.max_degree
              << " leaf_fraction=" << f.leaf_fraction
              << " isolated_fraction=" << f.isolated_fraction
              << " kcore_max=" << f.kcore_max
              << " kcore_mean=" << f.kcore_mean
              << " component_count=" << f.component_count
              << " largest_component_fraction=" << f.largest_component_fraction
              << " second_component_fraction=" << f.second_component_fraction
              << " clustering_sampled_mean=" << f.clustering_sampled_mean
              << " clustering_samples_used=" << f.clustering_samples_used
              << " io_time=" << io_time
              << " feature_time=" << feature_time
              << std::endl;

    std::cout << "GUESS"
              << " class=" << decision.guessed_class
              << " preset=" << decision.preset_name
              << " confidence=" << decision.confidence
              << " lp=" << decision.toggles.enable_lp
              << " trivial=" << decision.toggles.enable_trivial
              << " pr1=" << decision.toggles.enable_pr1
              << " pr2=" << decision.toggles.enable_pr2
              << " pr3=" << decision.toggles.enable_pr3
              << " pr4=" << decision.toggles.enable_pr4
              << " flags=\"" << autotune::disabledFlags(decision.toggles) << "\"";
    if (verbose) {
        std::cout << " rationale=\"" << decision.rationale << "\"";
    }
    std::cout << std::endl;

    return 0;
}
