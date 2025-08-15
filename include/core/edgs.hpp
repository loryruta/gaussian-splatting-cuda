#pragma once

#include <memory>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "core/dataset.hpp"
#include "core/iinit_strategy.hpp"

/// Implementation of "EDGS: Eliminating Densification for Efficient Convergence of 3DGS":
/// https://arxiv.org/abs/2504.13204
class EDGS : public IInitStrategy {
public:
    explicit EDGS(const gs::param::TrainingParameters& training_params, std::shared_ptr<CameraDataset> camera_dataset);
    EDGS(const EDGS&) = delete;
    EDGS(EDGS&&) = delete;

    SplatData initialize_splat_data() override;

private:
    /// Find dense correspondences between the two views by inferring a Feature Matching model (e.g. RoMa).
    /// Correspondences are then triangulated into an output pointcloud (means and colors).
    void find_dense_correspondences(const CameraExample& cam_a,
                                    const CameraExample& cam_b,
                                    std::vector<torch::Tensor>& out_means,
                                    std::vector<torch::Tensor>& out_colors);

    /// Initialize pairs of nearby cameras used when inferring the Feature Matching model.
    /// Greedy approach: pairs are created by taking the first \c num_neighbors_per_camera_ neighbors of every camera.
    void init_camera_pairs();

    void load_feature_matching_model();

    // ----------------------------------------------------------------
    // Variables
    // ----------------------------------------------------------------

    std::random_device random_device_; // TODO Init only one random device/generator for the whole application
    std::unique_ptr<std::mt19937> rng_;

    const gs::param::TrainingParameters& training_params_;
    const torch::Tensor scene_center_;
    const std::shared_ptr<CameraDataset> camera_dataset_;

    std::vector<uint64_t> camera_pairs_;

    torch::jit::script::Module feature_matching_model_;

    /* Parameters */
    bool save_debug_images_ = false;
    int num_neighbors_per_camera_ = 16; ///< Number of neighbors per camera to inspect when creating pairs.
    float certainty_threshold_ = 0.8f;  ///< Only warps with a certainty above this threshold are considered to build the pointcloud.
};