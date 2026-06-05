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
import itertools
from dust3r.utils.image import imread_cv2
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import pdb

class VirtualKITTI2(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.max_interval = 5
        super().__init__(*args, **kwargs)
        assert self.split is not None
        self._load_data(self.split)
    
    def _load_data(self, split=None):
        scene_dirs = sorted(
            [
                d
                for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
            ]
        )
        if split == "train":
            scene_dirs = scene_dirs[:-1]
        elif split == "test":
            scene_dirs = scene_dirs[-1:]
        seq_dirs = []
        for scene in scene_dirs:
            seq_dirs += sorted(
                [
                    os.path.join(scene, d)
                    for d in os.listdir(os.path.join(self.ROOT, scene))
                    if os.path.isdir(os.path.join(self.ROOT, scene, d))
                ]
            )
        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        for seq_idx, seq in enumerate(seq_dirs):
            seq_path = osp.join(self.ROOT, seq)
            for cam in ["Camera_0", "Camera_1"]:
                basenames = sorted(
                    [
                        f[:5]
                        for f in os.listdir(seq_path + "/" + cam)
                        if f.endswith(".jpg")
                    ]
                )
                num_imgs = len(basenames)
                cut_off = (
                    self.num_views
                )
                if num_imgs < cut_off:
                    print(f"Skipping {scene}")
                    continue
                img_ids = list(np.arange(num_imgs) + offset)
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                scenes.append(seq + "/" + cam)
                scene_img_list.append(img_ids)
                sceneids.extend([j] * num_imgs)
                images.extend(basenames)
                start_img_ids.extend(start_img_ids_)
                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)
    
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
    
    def _get_views(self, idx, resolution, rng):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            self.num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=0.9,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
    
        for v, view_idx in enumerate(image_idxs):
            # load the view (and use the next one if this one's broken)
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            basename = self.images[view_idx]

            img = basename + "_rgb.jpg"
            image = imread_cv2(osp.join(scene_dir, img))
            depthmap = (
                cv2.imread(
                    osp.join(scene_dir, basename + "_depth.png"),
                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                ).astype(np.float32)
                / 100.0
            )
            camera_params = np.load(osp.join(scene_dir, basename + "_cam.npz"))
            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, img)
            )

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='VirtualKITTI2',
                label=scene_dir,
                instance=scene_dir + "_" + img))
        return views
