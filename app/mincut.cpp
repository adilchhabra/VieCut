/******************************************************************************
 * mincut.cpp
 *
 * Source of VieCut
 *
 ******************************************************************************
 * Copyright (C) 2017-2018 Alexander Noe <alexander.noe@univie.ac.at>
 *
 * Published under the MIT license in the LICENSE file.
 *****************************************************************************/

#include <ext/alloc_traits.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "algorithms/global_mincut/algorithms.h"
#include "algorithms/global_mincut/minimum_cut.h"
#include "common/configuration.h"
#include "common/definitions.h"
#include "data_structure/graph_access.h"
#include "data_structure/mutable_graph.h"
#include "io/graph_io.h"
#include "tlx/cmdline_parser.hpp"
#include "tlx/logger.hpp"
#include "tools/graph_features.h"
#include "tools/jet_upper_bound.h"
#include "tools/preset_selector.h"
#include "tools/random_functions.h"
#include "tools/string.h"
#include "tools/timer.h"

// typedef graph_access graph_type;
typedef mutable_graph graph_type;
typedef std::shared_ptr<graph_type> GraphPtr;

int main(int argn, char** argv) {
    static constexpr bool debug = false;

    tlx::CmdlineParser cmdl;
    size_t num_iterations = 1;

    auto cfg = configuration::getConfig();
    bool disable_pr1 = false;
    bool disable_pr2 = false;
    bool disable_pr3 = false;
    bool disable_pr4 = false;
    bool disable_lp = false;
    bool disable_trivial = false;
    bool auto_preset = false;
    bool jet_ub = false;

    cmdl.add_param_string("graph", cfg->graph_filename, "path to graph file");
#ifdef PARALLEL
    std::vector<std::string> procs;
    cmdl.add_stringlist('p', "proc", procs, "number of processes");
#endif
    cmdl.add_param_string("algo", cfg->algorithm, "algorithm name");
    cmdl.add_string('q', "pq", cfg->queue_type,
                    "name of priority queue implementation");
    cmdl.add_size_t('i', "iter", num_iterations, "number of iterations");
    cmdl.add_bool('l', "disable_limiting", cfg->disable_limiting,
                  "disable limiting of PQ values");
    cmdl.add_bool('s', "save_cut", cfg->save_cut,
                  "find which vertices are on which side of minimum cut");
    cmdl.add_double('c', "contraction_factor", cfg->contraction_factor,
                    "contraction factor for pre-run of viecut");
    cmdl.add_string('k', "sampling_type", cfg->sampling_type,
                    "sampling variant for pre-run of viecut");
    cmdl.add_flag('b', "balanced", cfg->find_most_balanced_cut,
                  "find most balanced minimum cut");
    cmdl.add_flag('d', "minimize conductance", cfg->find_lowest_conductance,
                  "find lowest conductance minimum cut");
    cmdl.add_string('o', "output_path", cfg->output_path,
                    "print minimum cut to file");
    cmdl.add_flag('v', "verbose", cfg->verbose, "more verbose logs");
    cmdl.add_string('e', "edge_select", cfg->edge_selection, "NNI edge select");
    cmdl.add_size_t('r', "seed", cfg->seed, "random seed");
    cmdl.add_string('t', "cactus_filename", cfg->cactus_filename,
                    "name of GraphML file for the cactus graph");
    cmdl.add_flag('A', "disable_pr1", disable_pr1,
                  "disable Padberg-Rinaldi reduction rule 1");
    cmdl.add_flag('B', "disable_pr2", disable_pr2,
                  "disable Padberg-Rinaldi reduction rule 2");
    cmdl.add_flag('C', "disable_pr3", disable_pr3,
                  "disable Padberg-Rinaldi reduction rule 3");
    cmdl.add_flag('D', "disable_pr4", disable_pr4,
                  "disable Padberg-Rinaldi reduction rule 4");
    cmdl.add_flag('E', "disable_lp", disable_lp,
                  "disable label propagation contraction");
    cmdl.add_flag('F', "disable_trivial", disable_trivial,
                  "disable trivial-cut local search");
    cmdl.add_flag('X', "auto_preset", auto_preset,
                  "guess graph class/preset from fast feature scan");
    cmdl.add_flag('J', "jet_ub", jet_ub,
                  "use JET min-cut mode as an initial upper bound");
    cmdl.add_string('Y', "jet_config", cfg->jet_config_file,
                    "optional JET config file");
    cmdl.add_size_t('Z', "jet_iter", cfg->jet_num_iterations,
                    "number of JET upper-bound attempts");

    if (!cmdl.process(argn, argv))
        return -1;

    if (cfg->find_lowest_conductance) {
        // same check, just different optimization function, rest of code reused
        cfg->find_most_balanced_cut = true;
    }

    cfg->enable_pr1 = true;
    cfg->enable_pr2 = true;
    cfg->enable_pr3 = true;
    cfg->enable_pr4 = true;
    cfg->enable_label_propagation = true;
    cfg->enable_trivial_cut_search = true;

    if (cfg->cactus_filename != "" ) {
        // need save_cut to properly maintain containedVertices, see https://github.com/VieCut/VieCut/issues/7
	cfg->save_cut = true;
    }

    std::vector<int> numthreads;
    timer t;
    GraphPtr G = graph_io::readGraphWeighted<graph_type>(
        configuration::getConfig()->graph_filename);

    LOG1 << "io time: " << t.elapsed();

    cfg->use_jet_upper_bound = jet_ub;
    cfg->jet_upper_bound_available = false;
    if (cfg->use_jet_upper_bound) {
        timer jet_timer;
        EdgeWeight min_degree = G->getMinDegree();
        if (min_degree <= 1) {
            LOG1 << "JET_UB status=skipped"
                 << " graph=" << string::basename(cfg->graph_filename)
                 << " reason=min_degree_le_1"
                 << " jet_cut=NA"
                 << " min_degree=" << min_degree
                 << " used_cut=" << min_degree
                 << " improved_min_degree=0"
                 << " total_time=0"
                 << " conversion_time=0"
                 << " partition_time=0";
        } else {
            auto jet_result = jet_upper_bound::compute(G);
            if (jet_result.success) {
                cfg->initial_cut_upper_bound = jet_result.cut;
                cfg->jet_upper_bound_available = true;
                if (cfg->save_cut && jet_result.cut < min_degree &&
                    jet_result.partition.size() == G->number_of_nodes()) {
                    for (NodeID n : G->nodes()) {
                        G->setNodeInCut(n, jet_result.partition[n] != 0);
                    }
                }
                LOG1 << "JET_UB status=success"
                     << " graph=" << string::basename(cfg->graph_filename)
                     << " jet_cut=" << jet_result.cut
                     << " min_degree=" << min_degree
                     << " used_cut=" << std::min(min_degree, jet_result.cut)
                     << " improved_min_degree="
                     << (jet_result.cut < min_degree)
                     << " total_time=" << jet_result.time
                     << " conversion_time=" << jet_result.conversion_time
                     << " partition_time=" << jet_result.partition_time;
            } else {
                LOG1 << "JET_UB status=failed graph="
                     << string::basename(cfg->graph_filename)
                     << " message=\"" << jet_result.message << "\""
                     << " jet_cut=NA"
                     << " min_degree=" << min_degree
                     << " used_cut=" << min_degree
                     << " improved_min_degree=0"
                     << " total_time=" << jet_timer.elapsed()
                     << " conversion_time=" << jet_result.conversion_time
                     << " partition_time=" << jet_result.partition_time;
            }
        }
    }

    if (auto_preset) {
        auto feats = autotune::computeGraphFeatures(G);
        auto decision = autotune::recommendPreset(feats);
        autotune::applyPreset(cfg, decision.toggles);
        LOG1 << "AUTO_PRESET class=" << decision.guessed_class
             << " preset=" << decision.preset_name
             << " confidence=" << decision.confidence
             << " rationale=\"" << decision.rationale << "\""
             << " flags=\"" << autotune::disabledFlags(decision.toggles) << "\"";
    }

    // Explicit command-line disables always override auto preset choices.
    if (disable_pr1) cfg->enable_pr1 = false;
    if (disable_pr2) cfg->enable_pr2 = false;
    if (disable_pr3) cfg->enable_pr3 = false;
    if (disable_pr4) cfg->enable_pr4 = false;
    if (disable_lp) cfg->enable_label_propagation = false;
    if (disable_trivial) cfg->enable_trivial_cut_search = false;
    // ***************************** perform cut *****************************
#ifdef PARALLEL
    LOGC(cfg->verbose) << "PARALLEL DEFINED!";
    size_t i;
    try {
        for (i = 0; i < procs.size(); ++i) {
            numthreads.emplace_back(std::stoi(procs[i]));
        }
    } catch (...) {
        LOG1 << procs[i]
             << " is not a valid number of workers! Continuing without.";
    }
#else
    LOGC(cfg->verbose) << "PARALLEL NOT DEFINED";
#endif
    if (numthreads.empty())
        numthreads.emplace_back(1);
    timer tdegs;

    for (size_t i = 0; i < num_iterations; ++i) {
        for (int numthread : numthreads) {
            LOG << cfg->seed << " random seed";
            random_functions::setSeed(cfg->seed);

            NodeID n = G->number_of_nodes();
            EdgeID m = G->number_of_edges();

            auto mc = selectMincutAlgorithm<GraphPtr>(cfg->algorithm);
            omp_set_num_threads(numthread);
            cfg->threads = numthread;

            t.restart();
            EdgeWeight cut;
            cut = mc->perform_minimum_cut(G);

            if (cfg->output_path != "") {
                if (!cfg->save_cut) {
                    LOG1 << "Please enable -s to save cut. "
                         << "Otherwise it cannot be printed";
                    exit(1);
                }
                if (cfg->find_most_balanced_cut == false) {
                    // most balanced cut already prints inside of algorithm
                    graph_io::writeCut(G, cfg->output_path);
                }
            }

            std::string graphname = string::basename(cfg->graph_filename);
            std::string algprint = cfg->algorithm;
#ifdef PARALLEL
            algprint += "par";
#endif
            algprint += cfg->pq;

            if (cfg->disable_limiting) {
                algprint += "unlimited";
            }

            std::cout << "RESULT algo=" << algprint
                      << " graph=" << graphname
                      << " time=" << t.elapsed()
                      << " cut=" << cut
                      << " n=" << n
                      << " m=" << m / 2
                      << " processes=" << numthread
                      << " edge_select=" << cfg->edge_selection
                      << " seed=" << cfg->seed
                      << std::endl;
        }
    }
}
