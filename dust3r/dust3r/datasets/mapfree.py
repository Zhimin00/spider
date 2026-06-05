# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed habitat
# dataset at https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
# See datasets_preprocess/habitat for more details
# --------------------------------------------------------
import os.path as osp
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # noqa
import cv2  # noqa
import numpy as np
from PIL import Image
import json
import pickle
import h5py
from tqdm import tqdm
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import pdb
import itertools

class MapFree(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.max_interval = 30
        super().__init__(*args, **kwargs)
        assert self.split is not None
        self._load_data()

    def imgid2path(self, img_id, scene):
        first_seq_id, first_frame_id = img_id
        return os.path.join(
            self.ROOT,
            scene,
            f"dense{first_seq_id}",
            "rgb",
            f"frame_{first_frame_id:05d}.jpg",
        )

    def path2imgid(self, subscene, filename):
        first_seq_id = int(subscene[5:])
        first_frame_id = int(filename[6:-4])
        return [first_seq_id, first_frame_id]
    
    def _load_data(self):
        cache_file = f"{self.ROOT}/cached_metadata_50_col_only.h5"
        if os.path.exists(cache_file):
            print(f"Loading cached metadata from {cache_file}")
            with h5py.File(cache_file, "r") as hf:
                self.scenes = list(map(lambda x: x.decode("utf-8"), hf["scenes"][:]))
                self.sceneids = hf["sceneids"][:]
                self.scope = hf["scope"][:]
                self.video_flags = hf["video_flags"][:]
                self.groups = hf["groups"][:]
                self.id_ranges = hf["id_ranges"][:]
                self.images = hf["images"][:]
        else:
            scene_dirs = sorted(
                [
                    d
                    for d in os.listdir(self.ROOT)
                    if os.path.isdir(os.path.join(self.ROOT, d))
                ]
            )
            scenes = []
            sceneids = []
            groups = []
            scope = []
            images = []
            id_ranges = []
            is_video = []
            start = 0
            j = 0
            offset = 0

            for scene in tqdm(scene_dirs):
                scenes.append(scene)
                # video sequences
                subscenes = sorted(
                    [
                        d
                        for d in os.listdir(os.path.join(self.ROOT, scene))
                        if d.startswith("dense")
                    ]
                )
                id_range_subscenes = []
                for subscene in subscenes:
                    rgb_paths = sorted(
                        [
                            d
                            for d in os.listdir(
                                os.path.join(self.ROOT, scene, subscene, "rgb")
                            )
                            if d.endswith(".jpg")
                        ]
                    )
                    assert (
                        len(rgb_paths) > 0
                    ), f"{os.path.join(self.ROOT, scene, subscene)} is empty."
                    num_imgs = len(rgb_paths)
                    images.extend(
                        [self.path2imgid(subscene, rgb_path) for rgb_path in rgb_paths]
                    )
                    id_range_subscenes.append((offset, offset + num_imgs))
                    offset += num_imgs

                # image collections
                metadata = pickle.load(
                    open(os.path.join(self.ROOT, scene, "metadata.pkl"), "rb")
                )
                ref_imgs = list(metadata.keys())
                img_groups = []
                for ref_img in ref_imgs:
                    other_imgs = metadata[ref_img]
                    if len(other_imgs) + 1 < self.num_views:
                        continue
                    group = [(*other_img[0], other_img[1]) for other_img in other_imgs]
                    group.insert(0, (*ref_img, 1))
                    img_groups.append(np.array(group))
                    id_ranges.append(id_range_subscenes[ref_img[0]])
                    scope.append(start)
                    start = start + len(group)

                num_groups = len(img_groups)
                sceneids.extend([j] * num_groups)
                groups.extend(img_groups)
                is_video.extend([False] * num_groups)
                j += 1

            self.scenes = np.array(scenes)
            self.sceneids = np.array(sceneids)
            self.scope = np.array(scope)
            self.video_flags = np.array(is_video)
            self.groups = np.concatenate(groups, 0)
            self.id_ranges = np.array(id_ranges)
            self.images = np.array(images)

            data = dict(
                scenes=self.scenes,
                sceneids=self.sceneids,
                scope=self.scope,
                video_flags=self.video_flags,
                groups=self.groups,
                id_ranges=self.id_ranges,
                images=self.images,
            )

            with h5py.File(cache_file, "w") as h5f:
                h5f.create_dataset(
                    "scenes",
                    data=data["scenes"].astype(object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="lzf",
                    chunks=True,
                )
                h5f.create_dataset(
                    "sceneids", data=data["sceneids"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "scope", data=data["scope"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "video_flags",
                    data=data["video_flags"],
                    compression="lzf",
                    chunks=True,
                )
                h5f.create_dataset(
                    "groups", data=data["groups"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "id_ranges", data=data["id_ranges"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "images", data=data["images"], compression="lzf", chunks=True
                )

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video
    
    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list
        
    def _get_views(self, idx, resolution, rng):
        scene = self.scenes[self.sceneids[idx]]
        if rng.random() < 0.6:
            ids = np.arange(self.id_ranges[idx][0], self.id_ranges[idx][1])
            cut_off = self.num_views
            start_ids = ids[: len(ids) - cut_off + 1]
            start_id = rng.choice(start_ids)
            pos, ordered_video = self.get_seq_from_start_id(
                self.num_views,
                start_id,
                ids.tolist(),
                rng,
                max_interval=self.max_interval,
                video_prob=0.8,
                fix_interval_prob=0.5,
                block_shuffle=16,
            )
            ids = np.array(ids)[pos]
            image_idxs = self.images[ids]
        else:
            ordered_video = False
            seq_start_index = self.scope[idx]
            seq_end_index = self.scope[idx + 1] if idx < len(self.scope) - 1 else None
            image_idxs = (
                self.groups[seq_start_index:seq_end_index]
                if seq_end_index is not None
                else self.groups[seq_start_index:]
            )
            image_idxs, overlap_scores = image_idxs[:, :2], image_idxs[:, 2]
            replace = (
                True
                if len(overlap_scores[overlap_scores > 0]) < self.num_views
                else False
            )
            image_idxs = rng.choice(
                image_idxs,
                self.num_views,
                replace=replace,
                p=overlap_scores / np.sum(overlap_scores),
            )
            image_idxs = image_idxs.astype(np.int64)
        
        views = []
        
        for v, view_idx in enumerate(image_idxs):
            img_path = self.imgid2path(view_idx, scene)
            depth_path = img_path.replace("rgb", "depth").replace(".jpg", ".npy")
            cam_path = img_path.replace("rgb", "cam").replace(".jpg", ".npz")
            sky_mask_path = img_path.replace("rgb", "sky_mask")
            image = imread_cv2(img_path)
            depthmap = np.load(depth_path)
            camera_params = np.load(cam_path)
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_UNCHANGED) >= 127

            intrinsics = camera_params["intrinsic"].astype(np.float32)
            camera_pose = camera_params["pose"].astype(np.float32)

            depthmap[sky_mask] = -1.0
            depthmap[depthmap > 400.0] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(img_path)
            )

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MapFree',
                label=img_path,
                instance=img_path))
        return views

    def _load_one_view(self, data_path, key, view_index, resolution, rng):
        view_index += 1  # file indices starts at 1
        impath = osp.join(data_path, f"{key}_{view_index}.jpeg")
        image = Image.open(impath)

        depthmap_filename = osp.join(data_path, f"{key}_{view_index}_depth.exr")
        depthmap = cv2.imread(depthmap_filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

        camera_params_filename = osp.join(data_path, f"{key}_{view_index}_camera_params.json")
        with open(camera_params_filename, 'r') as f:
            camera_params = json.load(f)

        intrinsics = np.float32(camera_params['camera_intrinsics'])
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = camera_params['R_cam2world']
        camera_pose[:3, 3] = camera_params['t_cam2world']

        image, depthmap, intrinsics = self._crop_resize_if_necessary(
            image, depthmap, intrinsics, resolution, rng, info=impath)
        return image, depthmap, intrinsics, camera_pose
