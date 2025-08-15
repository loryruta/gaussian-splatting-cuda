#include "core/edgs.hpp"

#include <random>
#include <torch/script.h>

// Reference:
// https://arxiv.org/abs/2504.13204

EDGS::EDGS(const gs::param::TrainingParameters& training_params, std::shared_ptr<CameraDataset> camera_dataset) : training_params_(training_params),
                                                                                                                  camera_dataset_(camera_dataset) {
    rng_ = std::make_unique<std::mt19937>(random_device_());

    load_feature_matching_model();
    init_camera_pairs();
}

void EDGS::load_feature_matching_model() {
    const std::vector<std::string> default_paths = {
        "weights/feature_matching/roma_indoor.pt", // TODO roma_outdoor, tiny_roma_outdoor ... ?
        "../weights/feature_matching/roma_indoor.pt",
        "../../weights/feature_matching/roma_indoor.pt",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") + "/.cache/gaussian_splatting/feature_matching/roma_indoor.pt"};

    for (const auto& model_path : default_paths) {
        if (std::filesystem::exists(model_path)) {
            printf("[INFO ] [EDGS] Loading model \"%s\"\n", model_path.c_str());
            feature_matching_model_ = torch::jit::load(model_path);
            feature_matching_model_.eval();
            feature_matching_model_.to(torch::kCUDA);
            printf("[INFO ] [EDGS] \"%s\" model loaded\n", model_path.c_str());
            return;
        }
    }

    throw std::runtime_error("Feature matching model for EDGS not found.\n"
                             "Searched paths: weights/feature_matching/roma_indoor.pt, ../weights/feature_matching/roma_indoor.pt");
}

void EDGS::init_camera_pairs() {
    camera_pairs_.clear();

    const std::vector<std::shared_ptr<Camera>>& cameras = camera_dataset_->get_cameras();
    const int num_cameras = cameras.size();
    printf("[INFO ] [EDGS] Creating pairs with %d cameras...\n", num_cameras);

    camera_pairs_.resize(num_cameras, -1 /* Unassigned */);

    std::unordered_set<uint64_t> camera_pairs_hashset;

    torch::Tensor cam_locations = torch::empty({num_cameras, 16}); // (N, 16)
    for (int i = 0; i < cameras.size(); ++i) {
        cam_locations[i] = cameras.at(i)->world_view_transform().flatten();
    }

    for (int cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
        torch::Tensor cur_location = cam_locations[cam_idx];
        // Compute the Frobenius norm of the current camera to all others
        torch::Tensor distances = (cur_location - cam_locations).exp2().sum({-1}).sqrt(); // (N,)
        // Sort distances from lower to greater
        torch::Tensor sorted_indices = distances.argsort(false /* ascending */);
        assert(sorted_indices.numel() == 1);
        assert(sorted_indices.size(0) == num_cameras);
        assert(sorted_indices.scalar_type() == torch::ScalarType::Int);
        // Generate pairs
        // Stop criteria?
        // - When num_neighbors pairs have been generated (e.g. num_neighbors = 16)
        // - TODO Based on distance (to exclude far pairs)
        int num_paired = 0;
        for (int j = 1; j <= num_neighbors_per_camera_ && j < sorted_indices.size(0); ++j) { // Skip the first (current camera)
            int cam_neighbor_idx = sorted_indices[j].item<int>();
            uint64_t key; // Encode a key such that (a, b) = (b, a)
            if (cam_idx <= cam_neighbor_idx) {
                key = (uint64_t(cam_idx) << 32) | cam_neighbor_idx;
            } else {
                key = (uint64_t(cam_neighbor_idx) << 32) | cam_idx;
            }
            auto [iterator, inserted] = camera_pairs_hashset.emplace(key);
            if (inserted)
                ++num_paired;
        }
        printf("[DEBUG] [EDGS] Camera %d paired with %d neighboring cameras\n", cam_idx, num_paired);
    }

    printf("[INFO ] [EDGS] Created %zu camera pairs\n", camera_pairs_hashset.size());

    // Transfer the camera pairs hashset to a sorted vector. Sorted to improve visit efficiency
    camera_pairs_.reserve(camera_pairs_hashset.size());
    std::copy(camera_pairs_hashset.begin(), camera_pairs_hashset.end(), camera_pairs_.end());
    std::sort(camera_pairs_.begin(), camera_pairs_.end());
}

void EDGS::find_dense_correspondences(const CameraExample& cam_a,
                                      const CameraExample& cam_b,
                                      std::vector<torch::Tensor>& out_means,
                                      std::vector<torch::Tensor>& out_colors) {
    int W = 2; // TODO
    int H = 2; // TODO

    const Camera& cam_a_data = *cam_a.data.camera;
    const Camera& cam_b_data = *cam_b.data.camera;

    // ----------------------------------------------------------------
    // Infer the Feature Matching model (i.e. RoMa)
    // ----------------------------------------------------------------

    std::vector<torch::Tensor> forward_result = feature_matching_model_.forward({cam_a.target, cam_b.target}).toTensorVector();
    const torch::Tensor& warps = forward_result.at(0);     // (H, W, 4); [-1, 1] values
    const torch::Tensor& certainty = forward_result.at(1); // (H, W); [0, 1] values
    // Filter warps by their certainty
    torch::Tensor filtered_indices = (certainty >= certainty_threshold_).nonzero(); // (N, 2)
    const size_t num_filtered_warps = filtered_indices.size(0);
    if (num_filtered_warps == 0) {
        printf(R"([WARN] EDGS: Matching "%s" (UID %d) against "%s" (UID %d); all certainty values below the threshold %.2f (max is %.2f))",
               cam_a_data.image_name().c_str(),
               cam_a_data.uid(),
               cam_b_data.image_name().c_str(),
               cam_b_data.uid(),
               torch::max(certainty).item<float>(),
               certainty_threshold_);
        return;
    }
    std::vector<torch::indexing::TensorIndex> filtered_coords({filtered_indices.index({torch::indexing::Slice(), 0}),
                                                               filtered_indices.index({torch::indexing::Slice(), 1})});
    torch::Tensor filtered_warps = warps.index(filtered_coords); // (N, 4)
    torch::Tensor filtered_colors;                               // (N, 3)
    {
        auto filtered_colors_a = cam_a.target.index(filtered_coords);
        auto filtered_colors_b = cam_b.target.index(filtered_coords);
        filtered_colors = (filtered_colors_a + filtered_colors_b) * 0.5f;
    }

    // ----------------------------------------------------------------
    // Compute camera view-projection matrices
    // ----------------------------------------------------------------

    torch::Tensor P_a = cam_a_data.world_view_transform().matmul(cam_a_data.K()); // (3, 3); P_i in the paper
    torch::Tensor P_b = cam_b_data.world_view_transform().matmul(cam_b_data.K()); // (3, 3); P_j in the paper
    // torch matrices are row first!

    // ----------------------------------------------------------------
    // Construct A, b for least squares method (section 3.3)
    // ----------------------------------------------------------------

    // Obtain references to each individual component
    auto uv_a_u = filtered_warps.index({torch::indexing::Slice(), 0});
    auto uv_a_v = filtered_warps.index({torch::indexing::Slice(), 1});
    auto uv_b_u = filtered_warps.index({torch::indexing::Slice(), 2});
    auto uv_b_v = filtered_warps.index({torch::indexing::Slice(), 3});
    // Transform [-1,1] UV coordinates to screen space
    uv_a_u = (uv_a_u + 1.f) * 0.5f * W;
    uv_a_v = (uv_a_v + 1.f) * 0.5f * H;
    uv_b_u = (uv_b_u + 1.f) * 0.5f * W;
    uv_b_v = (uv_b_v + 1.f) * 0.5f * H;
    auto Pa_col0 = P_a.index({torch::indexing::Slice(), 0}); // (3,)
    auto Pa_col1 = P_a.index({torch::indexing::Slice(), 1}); // (3,)
    auto Pa_col2 = P_a.index({torch::indexing::Slice(), 2}); // (3,)
    auto Pb_col0 = P_b.index({torch::indexing::Slice(), 0}); // (3,)
    auto Pb_col1 = P_b.index({torch::indexing::Slice(), 1}); // (3,)
    auto Pb_col2 = P_b.index({torch::indexing::Slice(), 2}); // (3,)
    torch::Tensor A = torch::empty({4, 3}, torch::TensorOptions().dtype<float>().device(torch::kCUDA));
    A[0] = Pa_col0 - uv_a_u * Pa_col2; // A_{row,0}
    A[1] = Pa_col1 - uv_a_v * Pa_col2; // A_{row,1}
    A[2] = Pb_col0 - uv_b_u * Pb_col2; // A_{row,2}
    A[3] = Pb_col1 - uv_b_v * Pb_col2; // A_{row,3}
    torch::Tensor b = torch::zeros({4}, torch::TensorOptions().dtype<float>().device(torch::kCUDA));

    // ----------------------------------------------------------------
    // Run Least Squares method
    // ----------------------------------------------------------------

    auto [_3d_points, residuals, rank, singular_values] = torch::linalg_lstsq(A, b);
    assert(_3d_points.is_cuda());
    assert(_3d_points.ndimension() == 2 && _3d_points.size(0) == num_filtered_warps && _3d_points.size(1) == 3);

    // TODO filter by how close they are to the eq solution?

    out_means.push_back(_3d_points);
    out_colors.push_back(filtered_colors);
}

SplatData EDGS::initialize_splat_data() {
    std::vector<torch::Tensor> means_list;
    std::vector<torch::Tensor> colors_list;

    assert(!camera_pairs_.empty());
    assert(std::is_sorted(camera_pairs_.begin(), camera_pairs_.end()));

    CameraExample cur_cam_a;
    int cur_cam_a_idx = -1;
    for (uint64_t value : camera_pairs_) {
        int cam_a_idx = (int)((value >> 32) & 0xFF);
        int cam_b_idx = (int)(value & 0xFF);
        if (cur_cam_a_idx != cam_a_idx) { // Load and cache camera A (because camera_pairs_ is sorted)
            cur_cam_a = camera_dataset_->get(cur_cam_a_idx);
            cur_cam_a_idx = cam_a_idx;
        }
        // Load camera B everytime
        CameraExample cam_b = camera_dataset_->get(cam_b_idx);
        find_dense_correspondences(cur_cam_a, cam_b, means_list, colors_list);
    }

    PointCloud pcd{};
    pcd.means = torch::concat(means_list, 0);
    pcd.colors = torch::concat(colors_list, 0);
    return SplatData::init_model_from_pointcloud(training_params_, scene_center_, pcd);
}
