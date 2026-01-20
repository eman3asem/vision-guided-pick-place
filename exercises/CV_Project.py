import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
import random

def do_pose_estimation(scene_pointcloud, object_pointcloud):
    # print("YOU NEED TO IMPLEMENT THIS!")
    #1. Preprocess Point clouds
    scene_pointcloud_filtered= preprocess_pointcloud(scene_pointcloud)
    object_pointcloud= voxel_grid(object_pointcloud)
    #2. Global Pose Estimation
    global_pose=global_pose_estimation(copy.deepcopy(object_pointcloud), scene_pointcloud_filtered)
    alighned_object=copy.deepcopy(object_pointcloud)
    alighned_object.transform(global_pose)
    #3. Local Pose Estimation
    local_pose=local_pose_estimation(alighned_object, scene_pointcloud_filtered)
    #4. Combine poses to get the final pose
    final_pose = local_pose @ global_pose

    return final_pose

## ======= excercise 5 code below ======= ##
# This function just displays the effect of one of the functions visually
def display_removal(preserved_points, removed_points):
    removed_points.paint_uniform_color([1, 0, 0])        # Show removed points in red
    preserved_points.paint_uniform_color([0.8, 0.8, 0.8])# Show preserved points in gray
    # o3d.visualization.draw_geometries([removed_points, preserved_points])

def voxel_grid(input_cloud):
    voxel_down_cloud = input_cloud.voxel_down_sample(voxel_size=0.008)
    return voxel_down_cloud

def outlier_removal(input_cloud):
    cl, ind = input_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    # display_removal(input_cloud.select_by_index(ind), input_cloud.select_by_index(ind, invert=True))
    return input_cloud.select_by_index(ind)

def spatial_filter(input_cloud):
    # Define bounding box limits calculated from the scene
    min_bound = [-1.30, -0.25, 0.9]
    max_bound = [1.278, 0.15, 1.3]

    passthrough = input_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,
                                                                        max_bound=max_bound))
    # display_removal(passthrough, input_cloud)
    return passthrough

def preprocess_pointcloud(input_cloud):
    # o3d.visualization.draw_geometries_with_editing([input_cloud])
    cloud_filtered = voxel_grid(input_cloud)
    print(f'voxel grid {len(cloud_filtered.points)} points')
    cloud_filtered = outlier_removal(cloud_filtered)
    print(f'outlier removal {len(cloud_filtered.points)} points')
    cloud_filtered = spatial_filter(cloud_filtered)
    print(f'spatial filter {len(cloud_filtered.points)} points')
    cloud_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    
    return cloud_filtered

## ======= end of excercise 5 code ======= ##

### ======= excercise 7 code below ======= ##
########### local pose estimation ###########
def create_kdtree(scn):
    tree = o3d.geometry.KDTreeFlann(scn)
    return tree

def set_ICP_parameters():
    it = 100
    thressq = 0.012**2
    return it, thressq

def find_closest_points(obj_aligned, tree, thressq):
    # 1) Find closest points
    corr = o3d.utility.Vector2iVector()
    for j in range(len(obj_aligned.points)):
        k, idx, dist = tree.search_knn_vector_3d(obj_aligned.points[j], 1)
        
        # Apply distance threshold to correspondences
        if dist[0] < thressq:
            corr.append((j, idx[0]))
    return corr

def estimate_transformation(obj, scn, corr):
    # 2) Estimate transformation using point-to-plane metric for better accuracy
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    T = est.compute_transformation(obj, scn, corr)
    return T

def apply_pose(obj, T):
    # 3) Apply pose
    obj.transform(T)
    return obj

def update_result_pose_ICP(pose, T):
    # 4) Update result
    pose = T if pose is None else T @ pose
    return pose

def local_pose_estimation(obj, scn):
    # Create a k-d tree for scene
    tree = create_kdtree(scn)

    # Set ICP parameters
    it, thressq = set_ICP_parameters()

    # Start ICP
    pose = None
    obj_aligned = o3d.geometry.PointCloud(obj)
    for i in tqdm(range(it), desc='ICP'):
        # 1) Find closest points
        corr = find_closest_points(obj_aligned, tree, thressq)
            
        # 2) Estimate transformation
        T = estimate_transformation(obj_aligned, scn, corr)
        
        # 3) Apply pose
        obj_aligned = apply_pose(obj_aligned, T)
        
        # 4) Update result
        pose = update_result_pose_ICP(pose, T)

    # Print pose
    print('Got the following pose:')
    print(pose)

    # Apply pose to the original object
    obj = apply_pose(obj, pose)

    return pose

########## global pose estimation ##########
def set_RANSAC_parameters():
    it = 2000
    thressq = 0.01**2
    return it, thressq

def compute_surface_normals(obj, scn):
    obj.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.03))
    scn.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.03))
    return

def compute_shape_features(obj, scn):
    obj_features = o3d.pipelines.registration.compute_fpfh_feature(obj, search_param=o3d.geometry.KDTreeSearchParamRadius(0.05))
    scn_features = o3d.pipelines.registration.compute_fpfh_feature(scn, search_param=o3d.geometry.KDTreeSearchParamRadius(0.05))
    return obj_features, scn_features

def find_feature_matches(obj_features, scn_features):
    corr = o3d.utility.Vector2iVector()
    for j in tqdm(range(obj_features.shape[0]), desc='Correspondences'):
        fobj = obj_features[j]
        dist = np.sum((fobj - scn_features)**2, axis=-1)
        kmin = np.argmin(dist)
        corr.append((j, kmin))
    return corr

def create_kdtree(scn):
    tree = o3d.geometry.KDTreeFlann(scn)
    return tree

def apply_pose(obj, T):
    obj.transform(T)
    return obj

# To ensure that we sample 3 unique correspondences
def sample_3_random_correspondences(corr):
    # Check if we have enough correspondences
    if len(corr) < 3:
        return None

    # FIX suggested by AI: random.sample cannot handle Open3D vectors directly.
    # We sample 3 random indices from the range of the correspondence size.
    idx = random.sample(range(len(corr)), k=3)

    # Retrieve the specific correspondences using the sampled indices
    # and create a new Vector2iVector
    sampled_list = [corr[i] for i in idx]
    random_corr = o3d.utility.Vector2iVector(sampled_list)
    
    return random_corr

def estimate_transformation(obj, scn, corr):
    # Estimate transformation
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    T = est.compute_transformation(obj, scn, corr)
    return T

def validate(obj_aligned, tree, thressq):
    inliers = 0
    for j in range(len(obj_aligned.points)):
        k, idx, dist = tree.search_knn_vector_3d(obj_aligned.points[j], 1)
        if dist[0] < thressq:
            inliers += 1
    return inliers

def update_result_pose(pose_best, T, inliers, inliers_best, obj):
    if inliers > inliers_best:
        print(f'Got a new model with {inliers}/{len(obj.points)} inliers!')
        inliers_best = inliers
        pose_best = T
    else:
        pose_best = pose_best
        inliers_best = inliers_best
    return pose_best, inliers_best

def global_pose_estimation(obj, scn):
    # Set RANSAC parameters
    it, thressq = set_RANSAC_parameters()

    # Compute surface normals
    compute_surface_normals(obj, scn)

    # Compute shape features
    obj_features, scn_features = compute_shape_features(obj, scn)

    obj_features = np.asarray(obj_features.data).T
    scn_features = np.asarray(scn_features.data).T

    # Find feature matches
    corr = find_feature_matches(obj_features, scn_features)

    # Create a k-d tree for scene
    tree = create_kdtree(scn)

    # Start RANSAC
    random.seed(123456789)
    inliers_best = 0
    pose_best = None
    for i in tqdm(range(it), desc='RANSAC'):   
        # Sample 3 random correspondences
        corr_i = sample_3_random_correspondences(corr)
        
        # Estimate transformation
        T = estimate_transformation(obj, scn, corr_i)
        
        # Apply pose (to a copy of the object)
        obj_aligned = o3d.geometry.PointCloud(obj)
        obj_aligned = apply_pose(obj_aligned, T)
        
        # Validate
        inliers = validate(obj_aligned, tree, thressq)

        # Update result
        pose_best, inliers_best = update_result_pose(pose_best, T, inliers, inliers_best, obj_aligned)

    # Print pose
    print('Got the following pose:')
    print(pose_best)

    # Apply pose to the original object
    obj = apply_pose(obj, pose_best)

    return pose_best

### ======= excercise 7 code end ======= ##

