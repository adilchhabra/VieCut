/******************************************************************************
 * preset_selector.h
 *
 * Lightweight heuristic class/preset guesser from graph features.
 *
 *****************************************************************************/

#pragma once

#include <sstream>
#include <string>

#include "common/configuration.h"
#include "tools/graph_features.h"

namespace autotune {

struct PresetToggles {
    bool enable_lp = true;
    bool enable_trivial = true;
    bool enable_pr1 = true;
    bool enable_pr2 = true;
    bool enable_pr3 = true;
    bool enable_pr4 = true;
};

struct PresetDecision {
    std::string guessed_class = "unknown";
    std::string preset_name = "baseline";
    double confidence = 0.0;
    std::string rationale = "";
    PresetToggles toggles;
};

inline PresetToggles togglesForPreset(const std::string& preset) {
    PresetToggles t;
    if (preset == "no_lp") {
        t.enable_lp = false;
    } else if (preset == "no_trivial") {
        t.enable_trivial = false;
    } else if (preset == "no_pr1") {
        t.enable_pr1 = false;
    } else if (preset == "no_pr2") {
        t.enable_pr2 = false;
    } else if (preset == "no_pr3") {
        t.enable_pr3 = false;
    } else if (preset == "no_pr4") {
        t.enable_pr4 = false;
    }
    return t;
}

inline std::string disabledFlags(const PresetToggles& t) {
    std::ostringstream oss;
    if (!t.enable_pr1) oss << " -A";
    if (!t.enable_pr2) oss << " -B";
    if (!t.enable_pr3) oss << " -C";
    if (!t.enable_pr4) oss << " -D";
    if (!t.enable_lp) oss << " -E";
    if (!t.enable_trivial) oss << " -F";
    return oss.str();
}

inline PresetDecision recommendPreset(const GraphFeatures& f) {
    PresetDecision d;
    d.preset_name = "no_trivial";
    d.toggles = togglesForPreset(d.preset_name);
    d.confidence = 0.45;
    d.rationale = "default fast preset";

    if (f.nodes == 0) {
        d.preset_name = "baseline";
        d.toggles = togglesForPreset("baseline");
        d.confidence = 0.0;
        d.rationale = "empty graph";
        return d;
    }

    const bool sparse_road_like = (f.avg_degree <= 3.2 && f.leaf_fraction < 0.12);
    const bool very_dense_like = (f.avg_degree >= 30.0 || f.density >= 2e-3);
    const bool heavy_tail =
        (f.degree_p99 >= 200.0 && f.degree_cv >= 2.0);

    if (sparse_road_like) {
        d.guessed_class = "street";
        d.preset_name = "no_lp";
        d.toggles = togglesForPreset(d.preset_name);
        d.confidence = 0.67;
        d.rationale = "very sparse degree profile";
        return d;
    }

    if (very_dense_like) {
        d.guessed_class = "er";
        d.preset_name = "no_lp";
        d.toggles = togglesForPreset(d.preset_name);
        d.confidence = 0.70;
        d.rationale = "dense graph profile";
        return d;
    }

    if (heavy_tail) {
        d.guessed_class = "social";
        d.preset_name = "no_pr4";
        d.toggles = togglesForPreset(d.preset_name);
        d.confidence = 0.56;
        d.rationale = "heavy-tailed degree distribution";
        return d;
    }

    if (f.avg_degree >= 6.0 && f.degree_cv < 0.9) {
        d.guessed_class = "mesh";
        d.preset_name = "no_pr4";
        d.toggles = togglesForPreset(d.preset_name);
        d.confidence = 0.52;
        d.rationale = "uniform/mesh-like degree profile";
        return d;
    }

    d.guessed_class = "web";
    return d;
}

inline void applyPreset(const std::shared_ptr<configuration>& cfg,
                        const PresetToggles& t) {
    cfg->enable_label_propagation = t.enable_lp;
    cfg->enable_trivial_cut_search = t.enable_trivial;
    cfg->enable_pr1 = t.enable_pr1;
    cfg->enable_pr2 = t.enable_pr2;
    cfg->enable_pr3 = t.enable_pr3;
    cfg->enable_pr4 = t.enable_pr4;
}

}  // namespace autotune

