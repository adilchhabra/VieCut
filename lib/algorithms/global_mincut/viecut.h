/******************************************************************************
 * viecut.h
 *
 * Source of VieCut
 *
 ******************************************************************************
 * Copyright (C) 2017-2018 Alexander Noe <alexander.noe@univie.ac.at>
 *
 * Published under the MIT license in the LICENSE file.
 *****************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "algorithms/flow/excess_scaling.h"
#include "algorithms/global_mincut/minimum_cut.h"
#include "algorithms/global_mincut/minimum_cut_helpers.h"
#include "algorithms/global_mincut/noi_minimum_cut.h"
#include "algorithms/misc/strongly_connected_components.h"
#include "common/definitions.h"
#include "data_structure/flow_graph.h"
#include "data_structure/graph_access.h"
#include "tlx/logger.hpp"
#include "tools/graph_extractor.h"
#include "tools/timer.h"

#ifdef PARALLEL
#include "parallel/coarsening/contract_graph.h"
#include "parallel/coarsening/contraction_tests.h"
#include "parallel/coarsening/label_propagation.h"
#else
#include "coarsening/contract_graph.h"
#include "coarsening/contraction_tests.h"
#include "coarsening/label_propagation.h"
#endif

template <class GraphPtr>
class viecut : public minimum_cut {
 public:
    typedef GraphPtr GraphPtrType;
    static constexpr bool debug = false;
    bool timing = configuration::getConfig()->verbose;
    viecut() { }

    virtual ~viecut() { }

    EdgeWeight perform_minimum_cut(GraphPtr G) {
        return perform_minimum_cut(G, false);
    }

    EdgeWeight perform_minimum_cut(GraphPtr G,
                                   bool indirect) {
        if (!G) {
            return -1;
        }

        auto cfg = configuration::getConfig();
        auto emit_step = [&](const char* step,
                             NodeID n_before,
                             EdgeID m_before,
                             NodeID n_after,
                             EdgeID m_after,
                             EdgeWeight cut_before,
                             EdgeWeight cut_after,
                             double time_sec) {
            if (!timing) {
                return;
            }

            std::cout << "STEP"
                      << " step=" << step
                      << " graph=" << cfg->graph_filename
                      << " seed=" << cfg->seed
                      << " threads=" << cfg->threads
                      << " lp=" << cfg->enable_label_propagation
                      << " trivial=" << cfg->enable_trivial_cut_search
                      << " pr1=" << cfg->enable_pr1
                      << " pr2=" << cfg->enable_pr2
                      << " pr3=" << cfg->enable_pr3
                      << " pr4=" << cfg->enable_pr4
                      << " n_before=" << n_before
                      << " m_before=" << (m_before / 2)
                      << " n_after=" << n_after
                      << " m_after=" << (m_after / 2)
                      << " cut_before=" << cut_before
                      << " cut_after=" << cut_after
                      << " time=" << time_sec
                      << std::endl;
        };

        EdgeWeight cut = G->getMinDegree();
        std::vector<GraphPtr> graphs;
        graphs.push_back(G);

        minimum_cut_helpers<GraphPtr>::setInitialCutValues(graphs);
        const bool use_lp = cfg->enable_label_propagation;
        const bool use_trivial = cfg->enable_trivial_cut_search;

        while (graphs.back()->number_of_nodes() > 10000 &&
               (graphs.size() == 1 ||
                (graphs.back()->number_of_nodes() <
                 graphs[graphs.size() - 2]->number_of_nodes()))) {
            G = graphs.back();
            NodeID n_before = G->number_of_nodes();
            EdgeID m_before = G->number_of_edges();
            EdgeWeight cut_before = cut;
            std::vector<NodeID> mapping;
            std::vector<std::vector<NodeID> > reverse_mapping;
            double lp_time = 0.0;
            double trivial_time = 0.0;

            if (use_lp && use_trivial) {
                // Fast path: preserve existing behavior without extra checks.
                timer t_lp;
                label_propagation<GraphPtr> lp;
                std::vector<NodeID> cluster_mapping = lp.propagate_labels(G);
                auto remap = minimum_cut_helpers<GraphPtr>::remap_cluster(
                    G, cluster_mapping);
                mapping = std::move(remap.first);
                reverse_mapping = std::move(remap.second);
                lp_time = t_lp.elapsed();
                LOGC(timing) << "LP (total): " << lp_time;
                contraction::findTrivialCuts(G, &mapping, &reverse_mapping, cut);
                trivial_time = t_lp.elapsed() - lp_time;
                LOGC(timing) << "Trivial Cut Local Search: " << trivial_time;
            } else {
                if (use_lp) {
                    timer t_lp;
                    label_propagation<GraphPtr> lp;
                    std::vector<NodeID> cluster_mapping = lp.propagate_labels(G);
                    auto remap = minimum_cut_helpers<GraphPtr>::remap_cluster(
                        G, cluster_mapping);
                    mapping = std::move(remap.first);
                    reverse_mapping = std::move(remap.second);
                    lp_time = t_lp.elapsed();
                    LOGC(timing) << "LP (total): " << lp_time;
                } else {
                    mapping.resize(G->number_of_nodes());
                    reverse_mapping.resize(G->number_of_nodes());
                    for (NodeID n : G->nodes()) {
                        mapping[n] = n;
                        reverse_mapping[n].emplace_back(n);
                        if (configuration::getConfig()->save_cut) {
                            G->setPartitionIndex(n, n);
                        }
                    }
                    LOGC(timing) << "LP disabled: identity mapping";
                }
                emit_step("lp",
                          n_before, m_before,
                          n_before, m_before,
                          cut_before, cut_before,
                          lp_time);

                if (use_trivial) {
                    timer t_trivial;
                    contraction::findTrivialCuts(
                        G, &mapping, &reverse_mapping, cut);
                    trivial_time = t_trivial.elapsed();
                    LOGC(timing) << "Trivial Cut Local Search: "
                                 << trivial_time;
                } else {
                    LOGC(timing) << "Trivial Cut Local Search disabled";
                }
            }

            if (use_lp && use_trivial) {
                emit_step("lp",
                          n_before, m_before,
                          n_before, m_before,
                          cut_before, cut_before,
                          lp_time);
            }
            emit_step("trivial",
                      n_before, m_before,
                      n_before, m_before,
                      cut_before, cut_before,
                      trivial_time);

            timer t_contract;
            auto H = contraction::contractGraph(G, mapping, reverse_mapping);
            graphs.push_back(H);
            cut = minimum_cut_helpers<GraphPtr>::updateCut(graphs, cut);
            LOGC(timing) << "Graph Contraction (to "
                         << graphs.back()->number_of_nodes()
                         << " nodes): " << t_contract.elapsed();

            NodeID pr12_n_before = graphs.back()->number_of_nodes();
            EdgeID pr12_m_before = graphs.back()->number_of_edges();
            EdgeWeight pr12_cut_before = cut;
            timer t_pr12;
            union_find uf = tests::prTests12(graphs.back(), cut);
            graphs.push_back(
                contraction::fromUnionFind(graphs.back(), &uf, true));
            cut = minimum_cut_helpers<GraphPtr>::updateCut(graphs, cut);
            emit_step("pr12",
                      pr12_n_before, pr12_m_before,
                      graphs.back()->number_of_nodes(),
                      graphs.back()->number_of_edges(),
                      pr12_cut_before, cut,
                      t_pr12.elapsed());

            NodeID pr34_n_before = graphs.back()->number_of_nodes();
            EdgeID pr34_m_before = graphs.back()->number_of_edges();
            EdgeWeight pr34_cut_before = cut;
            timer t_pr34;
            union_find uf2 = tests::prTests34(graphs.back(), cut);
            graphs.push_back(
                contraction::fromUnionFind(graphs.back(), &uf2, true));
            cut = minimum_cut_helpers<GraphPtr>::updateCut(graphs, cut);
            emit_step("pr34",
                      pr34_n_before, pr34_m_before,
                      graphs.back()->number_of_nodes(),
                      graphs.back()->number_of_edges(),
                      pr34_cut_before, cut,
                      t_pr34.elapsed());
            LOGC(timing) << "Padberg-Rinaldi Tests (to "
                         << graphs.back()->number_of_nodes()
                         << " nodes): " << (t_pr12.elapsed() + t_pr34.elapsed());
        }

        if (graphs.back()->number_of_nodes() > 1) {
            timer t;
            NodeID n_before = graphs.back()->number_of_nodes();
            EdgeID m_before = graphs.back()->number_of_edges();
            EdgeWeight cut_before = cut;
            noi_minimum_cut<GraphPtr> noi;
            cut = std::min(cut, noi.perform_minimum_cut(graphs.back(), true));
            emit_step("noi_finalize",
                      n_before, m_before,
                      graphs.back()->number_of_nodes(),
                      graphs.back()->number_of_edges(),
                      cut_before, cut,
                      t.elapsed());

            LOGC(timing) << "Exact Algorithm:"
                         << t.elapsedToZero() << " deg: "
                         << graphs.back()->getMinDegree();
        }

        if (!indirect && configuration::getConfig()->save_cut)
            minimum_cut_helpers<GraphPtr>::retrieveMinimumCut(graphs);

        return cut;
    }
};
