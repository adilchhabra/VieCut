/******************************************************************************
 * graph_features.h
 *
 * Fast graph feature extraction for preset/class guessing.
 *
 *****************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "common/definitions.h"

namespace autotune {

struct GraphFeatures {
    NodeID nodes = 0;
    EdgeID edges_undirected = 0;
    double avg_degree = 0.0;
    double density = 0.0;
    double min_degree = 0.0;
    double max_degree = 0.0;
    double degree_p50 = 0.0;
    double degree_p90 = 0.0;
    double degree_p99 = 0.0;
    double degree_stddev = 0.0;
    double degree_cv = 0.0;
    double leaf_fraction = 0.0;
    double isolated_fraction = 0.0;
    double degree_ratio_p99_p50 = 0.0;
    double degree_ratio_max_p90 = 0.0;
    double degree_tail_hill_alpha = 0.0;
    double kcore_max = 0.0;
    double kcore_mean = 0.0;
    double component_count = 0.0;
    double largest_component_fraction = 0.0;
    double second_component_fraction = 0.0;
    double clustering_sampled_mean = 0.0;
    double clustering_samples_used = 0.0;
};

inline double quantile_sorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return 0.0;
    }
    if (sorted.size() == 1) {
        return sorted[0];
    }
    if (q < 0.0) {
        q = 0.0;
    }
    if (q > 1.0) {
        q = 1.0;
    }
    double pos = q * static_cast<double>(sorted.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return sorted[lo];
    }
    double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

template <class GraphPtr>
GraphFeatures computeGraphFeatures(GraphPtr G) {
    static constexpr size_t kClusteringSampleBudget = 64;
    static constexpr EdgeID kClusteringDegreeCap = 20000;
    static constexpr size_t kTailTopBudget = 128;

    GraphFeatures f;
    if (!G) {
        return f;
    }

    f.nodes = G->number_of_nodes();
    f.edges_undirected = G->number_of_edges() / 2;
    if (f.nodes == 0) {
        return f;
    }

    std::vector<NodeID> degree_int;
    degree_int.reserve(f.nodes);
    std::vector<double> degree_values;
    degree_values.reserve(f.nodes);

    double degree_sum = 0.0;
    size_t leaves = 0;
    size_t isolated = 0;

    for (NodeID n : G->nodes()) {
        EdgeID deg_e = G->getUnweightedNodeDegree(n);
        double deg = static_cast<double>(deg_e);

        degree_values.push_back(deg);
        degree_int.push_back(static_cast<NodeID>(deg_e));
        degree_sum += deg;
        if (deg_e == 0) {
            ++isolated;
        } else if (deg_e == 1) {
            ++leaves;
        }
    }

    std::sort(degree_values.begin(), degree_values.end());

    f.avg_degree = degree_sum / static_cast<double>(f.nodes);
    f.min_degree = degree_values.front();
    f.max_degree = degree_values.back();
    f.degree_p50 = quantile_sorted(degree_values, 0.50);
    f.degree_p90 = quantile_sorted(degree_values, 0.90);
    f.degree_p99 = quantile_sorted(degree_values, 0.99);
    f.leaf_fraction = static_cast<double>(leaves) / static_cast<double>(f.nodes);
    f.isolated_fraction =
        static_cast<double>(isolated) / static_cast<double>(f.nodes);
    f.degree_ratio_p99_p50 =
        (f.degree_p50 > 0.0) ? (f.degree_p99 / f.degree_p50) : 0.0;
    f.degree_ratio_max_p90 =
        (f.degree_p90 > 0.0) ? (f.max_degree / f.degree_p90) : 0.0;

    double sq_sum = 0.0;
    for (double d : degree_values) {
        double centered = d - f.avg_degree;
        sq_sum += centered * centered;
    }
    f.degree_stddev = std::sqrt(sq_sum / static_cast<double>(f.nodes));
    f.degree_cv = (f.avg_degree > 0.0) ? (f.degree_stddev / f.avg_degree) : 0.0;

    if (f.nodes > 1) {
        double n = static_cast<double>(f.nodes);
        f.density = (2.0 * static_cast<double>(f.edges_undirected)) / (n * (n - 1.0));
    }

    // Hill-estimator-style heavy-tail proxy from top positive degrees.
    std::vector<double> positive_degrees;
    positive_degrees.reserve(degree_values.size());
    for (double d : degree_values) {
        if (d > 0.0) {
            positive_degrees.push_back(d);
        }
    }
    if (positive_degrees.size() >= 4) {
        std::sort(positive_degrees.begin(), positive_degrees.end(),
                  std::greater<double>());
        size_t k = std::min(kTailTopBudget, positive_degrees.size());
        double x_min = positive_degrees[k - 1];
        if (x_min > 0.0) {
            double sum_logs = 0.0;
            size_t used = 0;
            for (size_t i = 0; i < k; ++i) {
                if (positive_degrees[i] >= x_min) {
                    sum_logs += std::log(positive_degrees[i] / x_min);
                    ++used;
                }
            }
            if (used > 0 && sum_logs > 1e-12) {
                f.degree_tail_hill_alpha = static_cast<double>(used) / sum_logs;
            }
        }
    }

    // Connected component profile.
    {
        std::vector<uint8_t> seen(f.nodes, 0);
        std::vector<NodeID> queue;
        queue.reserve(f.nodes);

        NodeID largest = 0;
        NodeID second = 0;
        NodeID components = 0;

        for (NodeID start : G->nodes()) {
            if (seen[start]) {
                continue;
            }
            ++components;
            seen[start] = 1;
            queue.clear();
            queue.push_back(start);
            size_t head = 0;
            NodeID comp_size = 0;

            while (head < queue.size()) {
                NodeID v = queue[head++];
                ++comp_size;
                for (EdgeID e : G->edges_of(v)) {
                    NodeID tgt = G->getEdgeTarget(v, e);
                    if (!seen[tgt]) {
                        seen[tgt] = 1;
                        queue.push_back(tgt);
                    }
                }
            }

            if (comp_size >= largest) {
                second = largest;
                largest = comp_size;
            } else if (comp_size > second) {
                second = comp_size;
            }
        }

        f.component_count = static_cast<double>(components);
        f.largest_component_fraction =
            static_cast<double>(largest) / static_cast<double>(f.nodes);
        f.second_component_fraction =
            static_cast<double>(second) / static_cast<double>(f.nodes);
    }

    // Batagelj-Zaversnik k-core decomposition.
    {
        NodeID max_degree = 0;
        for (NodeID d : degree_int) {
            max_degree = std::max(max_degree, d);
        }

        std::vector<NodeID> degree = degree_int;
        std::vector<NodeID> position(f.nodes, 0);
        std::vector<NodeID> vertices(f.nodes, 0);
        std::vector<NodeID> buckets(static_cast<size_t>(max_degree) + 1, 0);

        for (NodeID d : degree) {
            ++buckets[d];
        }

        NodeID start = 0;
        for (size_t i = 0; i < buckets.size(); ++i) {
            NodeID num = buckets[i];
            buckets[i] = start;
            start += num;
        }

        for (NodeID v : G->nodes()) {
            NodeID d = degree[v];
            position[v] = buckets[d];
            vertices[position[v]] = v;
            ++buckets[d];
        }

        for (size_t i = buckets.size(); i > 1; --i) {
            buckets[i - 1] = buckets[i - 2];
        }
        buckets[0] = 0;

        for (NodeID idx = 0; idx < f.nodes; ++idx) {
            NodeID v = vertices[idx];
            for (EdgeID e : G->edges_of(v)) {
                NodeID u = G->getEdgeTarget(v, e);
                if (degree[u] > degree[v]) {
                    NodeID du = degree[u];
                    NodeID pu = position[u];
                    NodeID pw = buckets[du];
                    NodeID w = vertices[pw];
                    if (u != w) {
                        position[u] = pw;
                        position[w] = pu;
                        vertices[pu] = w;
                        vertices[pw] = u;
                    }
                    ++buckets[du];
                    --degree[u];
                }
            }
        }

        NodeID kmax = 0;
        double ksum = 0.0;
        for (NodeID d : degree) {
            kmax = std::max(kmax, d);
            ksum += static_cast<double>(d);
        }
        f.kcore_max = static_cast<double>(kmax);
        f.kcore_mean = ksum / static_cast<double>(f.nodes);
    }

    // Sampled local clustering coefficient.
    {
        std::vector<NodeID> candidates;
        candidates.reserve(f.nodes);
        for (NodeID v : G->nodes()) {
            if (degree_int[v] >= 2 && degree_int[v] <= kClusteringDegreeCap) {
                candidates.push_back(v);
            }
        }

        size_t samples =
            std::min(kClusteringSampleBudget, candidates.size());
        if (samples > 0) {
            std::vector<NodeID> sampled_nodes;
            sampled_nodes.reserve(samples);

            if (candidates.size() == samples) {
                sampled_nodes = candidates;
            } else {
                for (size_t i = 0; i < samples; ++i) {
                    size_t idx =
                        (i * candidates.size()) / samples;
                    if (idx >= candidates.size()) {
                        idx = candidates.size() - 1;
                    }
                    sampled_nodes.push_back(candidates[idx]);
                }
            }

            std::vector<uint32_t> marker(f.nodes, 0);
            uint32_t epoch = 1;
            std::vector<NodeID> neighbors;

            double sum_cc = 0.0;
            size_t used = 0;
            for (NodeID v : sampled_nodes) {
                if (epoch == std::numeric_limits<uint32_t>::max()) {
                    std::fill(marker.begin(), marker.end(), 0);
                    epoch = 1;
                }
                ++epoch;

                neighbors.clear();
                neighbors.reserve(degree_int[v]);
                for (EdgeID e : G->edges_of(v)) {
                    NodeID u = G->getEdgeTarget(v, e);
                    marker[u] = epoch;
                    neighbors.push_back(u);
                }

                const double d = static_cast<double>(neighbors.size());
                if (d < 2.0) {
                    continue;
                }

                uint64_t links_times_two = 0;
                for (NodeID u : neighbors) {
                    for (EdgeID ue : G->edges_of(u)) {
                        NodeID w = G->getEdgeTarget(u, ue);
                        if (marker[w] == epoch) {
                            ++links_times_two;
                        }
                    }
                }

                const double wedges = d * (d - 1.0);
                if (wedges > 0.0) {
                    sum_cc += static_cast<double>(links_times_two) / wedges;
                    ++used;
                }
            }

            if (used > 0) {
                f.clustering_sampled_mean =
                    sum_cc / static_cast<double>(used);
                f.clustering_samples_used = static_cast<double>(used);
            }
        }
    }

    return f;
}

}  // namespace autotune
