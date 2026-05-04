/******************************************************************************
 * noigen.cpp
 *
 * NOIGEN graph generator for VieCut.
 *****************************************************************************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/configuration.h"
#include "common/definitions.h"
#include "data_structure/mutable_graph.h"
#include "io/graph_io.h"
#include "tlx/cmdline_parser.hpp"
#include "tlx/logger.hpp"
#include "tools/random_functions.h"

namespace {

constexpr EdgeWeight kWeightScale = 1000000;

uint64_t edgeKey(NodeID u, NodeID v) {
    if (u > v) {
        std::swap(u, v);
    }
    return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
}

EdgeWeight drawCapacity() {
    const double sampled = random_functions::nextDouble(0.0, 100.0);
    return static_cast<EdgeWeight>(std::llround(sampled * kWeightScale));
}

double computeDensityPercent(uint64_t n, uint64_t m) {
    const long double max_edges =
        static_cast<long double>(n) * static_cast<long double>(n - 1) / 2.0L;
    return static_cast<double>(
        100.0L * static_cast<long double>(m) / max_edges);
}

void writeClusters(const std::vector<NodeID>& cluster,
                   const std::string& path) {
    std::ofstream out(path);
    for (NodeID v = 0; v < cluster.size(); ++v) {
        out << v << " " << cluster[v] << "\n";
    }
}

}  // namespace

int main(int argn, char** argv) {
    tlx::CmdlineParser cmdl;
    auto cfg = configuration::getConfig();

    size_t n = 0;
    size_t k = 1;
    size_t m_input = 0;
    double density = 0.0;
    double p = -1.0;
    bool use_edges = false;
    bool use_density = false;
    bool unbalanced_clusters = false;
    std::string output_path;
    std::string cluster_output_path;

    cmdl.add_param_size_t("n", n, "number of vertices");
    cmdl.add_param_string("output_path", output_path,
                          "output graph path in weighted METIS format");
    cmdl.add_size_t('k', "clusters", k, "number of clusters");
    cmdl.add_size_t('m', "edges", m_input,
                    "number of undirected edges; overrides density if set");
    cmdl.add_double('d', "density", density,
                    "edge density in percent, 0 < d <= 100");
    cmdl.add_double('p', "inter_scale", p,
                    "inter-cluster capacity scaling factor (default: 1/n)");
    cmdl.add_size_t('s', "seed", cfg->seed, "random seed");
    cmdl.add_string('c', "cluster_output", cluster_output_path,
                    "optional path to save cluster assignments");
    cmdl.add_flag('u', "unbalanced_clusters", unbalanced_clusters,
                  "assign clusters independently instead of balanced shuffle");

    if (!cmdl.process(argn, argv)) {
        return -1;
    }

    use_edges = m_input > 0;
    use_density = density > 0.0;

    if (n < 2) {
        LOG1 << "n must be at least 2";
        return -1;
    }
    if (k < 1 || k > n) {
        LOG1 << "k must satisfy 1 <= k <= n";
        return -1;
    }
    if (use_edges == use_density) {
        LOG1 << "Specify exactly one of --edges or --density";
        return -1;
    }

    const uint64_t max_edges =
        static_cast<uint64_t>(n) * static_cast<uint64_t>(n - 1) / 2ULL;
    uint64_t m = 0;
    if (use_edges) {
        m = m_input;
        density = computeDensityPercent(n, m);
    } else {
        if (density <= 0.0 || density > 100.0) {
            LOG1 << "density must satisfy 0 < d <= 100";
            return -1;
        }
        const long double target =
            static_cast<long double>(n) * static_cast<long double>(n - 1)
            * static_cast<long double>(density) / 200.0L;
        m = static_cast<uint64_t>(std::floor(target));
    }

    if (m < static_cast<uint64_t>(n - 1) || m > max_edges) {
        LOG1 << "Invalid edge count m=" << m
             << ". Need n-1 <= m <= n(n-1)/2";
        return -1;
    }

    if (p < 0.0) {
        p = 1.0 / static_cast<double>(n);
    }
    if (p <= 0.0 || p > 1.0) {
        LOG1 << "p must satisfy 0 < p <= 1";
        return -1;
    }

    random_functions::setSeed(static_cast<int>(cfg->seed));

    LOG1 << "NOIGEN n=" << n
         << " m=" << m
         << " density=" << density
         << " k=" << k
         << " p=" << p
         << " seed=" << random_functions::getSeed()
         << " cluster_mode=" << (unbalanced_clusters ? "unbalanced" : "balanced")
         << " weight_scale=" << kWeightScale;

    auto G = std::make_shared<mutable_graph>();
    G->start_construction(static_cast<NodeID>(n), static_cast<EdgeID>(2 * m));

    std::unordered_set<uint64_t> present_edges;
    present_edges.reserve(static_cast<size_t>(m * 1.3));

    std::vector<std::tuple<NodeID, NodeID, EdgeWeight> > edges;
    edges.reserve(m);

    auto add_edge = [&](NodeID u, NodeID v) {
        if (u == v) {
            return false;
        }
        const uint64_t key = edgeKey(u, v);
        if (!present_edges.emplace(key).second) {
            return false;
        }
        EdgeWeight w = drawCapacity();
        edges.emplace_back(std::min(u, v), std::max(u, v), w);
        return true;
    };

    for (NodeID v = 0; v + 1 < n; ++v) {
        add_edge(v, v + 1);
    }

    while (edges.size() < m) {
        const NodeID u = random_functions::nextInt(0, static_cast<unsigned>(n - 1));
        const NodeID v = random_functions::nextInt(0, static_cast<unsigned>(n - 1));
        add_edge(u, v);
    }

    std::vector<NodeID> cluster(n, 0);
    if (k > 1) {
        if (unbalanced_clusters) {
            for (NodeID v = 0; v < n; ++v) {
                cluster[v] = random_functions::nextInt(
                    0, static_cast<unsigned>(k - 1));
            }
        } else {
            std::vector<NodeID> perm(n);
            random_functions::permutate_vector_good(&perm, true);
            for (NodeID i = 0; i < n; ++i) {
                cluster[perm[i]] = i % k;
            }
        }
    }

    for (auto& edge : edges) {
        NodeID u = std::get<0>(edge);
        NodeID v = std::get<1>(edge);
        EdgeWeight w = std::get<2>(edge);
        if (k > 1 && cluster[u] != cluster[v]) {
            w = static_cast<EdgeWeight>(std::llround(p * static_cast<double>(w)));
        }
        G->new_edge(u, v, w);
    }
    G->finish_construction();

    graph_io::writeGraphWeighted(G, output_path);

    if (!cluster_output_path.empty()) {
        writeClusters(cluster, cluster_output_path);
    }

    return 0;
}
