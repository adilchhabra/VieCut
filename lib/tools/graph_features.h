/******************************************************************************
 * graph_features.h
 *
 * Fast graph feature extraction for preset/class guessing.
 *
 *****************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>
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
    GraphFeatures f;
    if (!G) {
        return f;
    }

    f.nodes = G->number_of_nodes();
    f.edges_undirected = G->number_of_edges() / 2;
    if (f.nodes == 0) {
        return f;
    }

    std::vector<double> degree_values;
    degree_values.reserve(f.nodes);

    double degree_sum = 0.0;
    size_t leaves = 0;
    size_t isolated = 0;

    for (NodeID n : G->nodes()) {
        double deg = 0.0;
        for (EdgeID e : G->edges_of(n)) {
            (void)e;
            deg += 1.0;
        }

        degree_values.push_back(deg);
        degree_sum += deg;
        if (deg == 0.0) {
            ++isolated;
        } else if (deg == 1.0) {
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

    return f;
}

}  // namespace autotune
