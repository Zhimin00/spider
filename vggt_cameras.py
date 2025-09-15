# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import trimesh
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import copy
import requests
import pdb
sys.path.append("/cis/home/zshao14/Downloads/vggt/")

from visual_util import integrate_camera_into_scene, apply_scene_alignment
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list
    # pdb.set_trace()
    # gt_rel = np.array([[ 0.93643, -0.14401,  0.31994, -0.74714], [ 0.20728,  0.96281, -0.17332, 0.90215], [-0.28308,  0.22862,  0.93145, -1.50572]])
    # new_extrinsics = predictions['extrinsic']
    # pred_extrinsic_cambranch = new_extrinsics[1]
    # new_extrinsics[1] = gt_rel
    # predictions.update({'extrinsic': new_extrinsics})
    # pdb.set_trace()
    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions#, pred_extrinsic_cambranch

def my_predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
    gt_rel = None,
    pred_rel = None,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_sky (bool): Apply sky segmentation mask (default: False)
        target_dir (str): Output directory for intermediate files (default: None)
        prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    gt_extrinsic = gt_rel @ extrinsics_matrices[0]
    pred_extrinsic = pred_rel @ extrinsics_matrices[0]
    n = np.linalg.norm(extrinsics_matrices[1, :3, 3])
    gt_t = gt_extrinsic[:3, 3]
    pred_t = pred_extrinsic[:3, 3]
    gt_t_new = gt_t / np.linalg.norm(gt_t) * n
    pred_t_new = pred_t / np.linalg.norm(pred_t) * n
    gt_extrinsic[:3, 3] = gt_t_new
    pred_extrinsic[:3, 3] = pred_t_new

    extrinsics_matrices = np.vstack([extrinsics_matrices, [gt_extrinsic, pred_extrinsic]])
    num_cameras = num_cameras + 2
    # pdb.set_trace()

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            if i ==1 :
                rgba_color = colormap(3 / num_cameras)
            elif i == 3:
                rgba_color = colormap(1 / num_cameras)
            else:
                rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            # pdb.set_trace()

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


target_dir = '/cis/home/zshao14/Documents/vggt/scannet/766'
with torch.no_grad():
    predictions = run_model(target_dir, model)
    #, pred_extrinsic_cambranch
gt_rel = np.array([[ 0.93643, -0.14401,  0.31994, -0.74714], 
                   [ 0.20728,  0.96281, -0.17332, 0.90215], 
                   [-0.28308,  0.22862,  0.93145, -1.50572], 
                   [0, 0, 0, 1]])
gt_cam2world = np.linalg.inv(gt_rel)
pred_rel = np.array([[ 0.97070002, -0.23848207,  0.02945809, 0.25891936], [ 0.23166279,  0.96133465,  0.14888939, -0.20267322], [-0.06382653, -0.13770258,  0.98841498, -0.94439628], [0, 0, 0, 1]])
# pred_rel = np.array([[ 0.94015303, -0.08140332,  0.33088636, -0.35849336],
#                      [ 0.1898143 ,  0.93154667, -0.31014727,  0.64076021],
#                      [-0.28298907,  0.35439286,  0.89124794, -0.67890269],
#                      [0, 0, 0, 1]])
pred_cam2world = np.linalg.inv(pred_rel)
# pred_extrinsic_cambranch = np.vstack([pred_extrinsic_cambranch,[0,0,0,1]])
glbscene = my_predictions_to_glb(
        predictions,
        conf_thres=50,
        filter_by_frames='all',
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=target_dir,
        prediction_mode='DepthandCamera',
        gt_rel=gt_rel,
        pred_rel=pred_rel
    )

glbscene.export('/cis/home/zshao14/Documents/vggt/scannet/766/vggt-cameras.glb')

