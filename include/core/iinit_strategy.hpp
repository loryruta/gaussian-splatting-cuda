#pragma once

#include "core/dataset.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"

/// An initialization strategy.
/// That is the algorithm instructed to populate gaussians before optimization and densification.
class IInitStrategy {
public:
    explicit IInitStrategy() = default;
    virtual ~IInitStrategy() = default;

    virtual SplatData initialize_splat_data() = 0;
};
