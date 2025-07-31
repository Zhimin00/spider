
import torch
import numpy as np

import pdb
import spider.utils.path_to_dust3r
from dust3r.utils.device import to_numpy, todevice
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud, depthmap_to_absolute_camera_coordinates
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import imread_cv2
import dust3r.datasets.utils.cropping as cropping
import tqdm

from spider.model import SPIDER_POINTMAP
import os.path as osp
import PIL
from PIL import Image
import cv2
import json

def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None

def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]

class Habitat_data:
    def __init__(self, size, ROOT, split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None):
        self.ROOT = ROOT
        self.num_views = 2
        self.split = split
        self._set_resolutions(resolution)

        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed

        assert self.split is not None
        # loading list of scenes
        with open(osp.join('/cis/home/zshao14/datasets', f'Habitat_{size}_scenes_{self.split}.txt')) as f:
            self.scenes = f.read().splitlines()
        self.instances = list(range(1, 5))

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '[' + ';'.join(f'{w}x{h}' for w, h in self._resolutions) + ']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')      

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def filter_scene(self, label, instance=None):
        if instance:
            subscene, instance = instance.split('_')
            label += '/' + subscene
            self.instances = [int(instance) - 1]
        valid = np.bool_([scene.startswith(label) for scene in self.scenes])
        assert sum(valid), 'no scene was selected for {label=} {instance=}'
        self.scenes = [scene for i, scene in enumerate(self.scenes) if valid[i]]

    def _get_views(self, idx, resolution, rng):
        scene = self.scenes[idx]
        data_path, key = osp.split(osp.join(self.ROOT, scene))
        views = []
        max_view_index = 5
        while not osp.isfile(osp.join(data_path, f"{key}_{max_view_index}.jpeg")):
            max_view_index = max_view_index - 1
            print('no view', max_view_index)
        two_random_views = [0, rng.choice(list(range(1, max_view_index)))]
        for view_index in two_random_views:
            # load the view (and use the next one if this one's broken)
            for ii in range(view_index, view_index + 5):
                image, depthmap, intrinsics, camera_pose = self._load_one_view(data_path, key, ii % 5, resolution, rng)
                if np.isfinite(camera_pose).all():
                    break
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Habitat',
                label=osp.relpath(data_path, self.ROOT),
                instance=f"{key}_{view_index}"))
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

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        # assert min_margin_x > W/5, f'Bad principal point in view={info}'
        # assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        # target_resolution = np.array(resolution)
        # image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        # last thing done!
        for view in views:
            # transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views
    


class Aerial_MegaDepth_data:
    def __init__(self, ROOT, split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None):
        self.ROOT = ROOT
        self.num_views = 2
        self.split = split
        self._set_resolutions(resolution)

        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed

        assert self.split is not None
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        if split == 'train1':
            with np.load(osp.join(self.ROOT, 'aerial_megadepth_train_part1.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']
        elif split == 'train2':
            with np.load(osp.join(self.ROOT, 'aerial_megadepth_train_part2.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']

        elif split == 'val':
            with np.load(osp.join(self.ROOT, 'aerial_megadepth_val.npz'), allow_pickle=True) as data:
                self.all_scenes = data['scenes']
                self.all_images = data['images']
                self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def __repr__(self):
        resolutions_str = '[' + ';'.join(f'{w}x{h}' for w, h in self._resolutions) + ']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')      

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]

        scene = self.all_scenes[scene_id]
        seq_path = osp.join(self.ROOT, scene)

        views = []

        for im_id in [im1_id, im2_id]:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + '.jpg'))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f'cannot load {img}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Aerial-MegaDepth',
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views
    
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        # assert min_margin_x > W/5, f'Bad principal point in view={info}'
        # assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        # target_resolution = np.array(resolution)
        # image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

        # last thing done!
        for view in views:
            # transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

def iter_views(views, device='numpy'):
    if device:
        views = todevice(views, device)
    assert views['img'].ndim == 4
    B = len(views['img'])
    for i in range(B):
        view = {k:(v[i] if isinstance(v, (np.ndarray,torch.Tensor)) else v) for k,v in views.items()}
        yield view


def gen_rel_pose(views, norm=True):
    assert len(views) == 2
    cam1_to_w, cam2_to_w = [view['camera_pose'] for view in views]
    w_to_cam1 = np.linalg.inv(cam1_to_w)

    cam2_to_cam1 = w_to_cam1 @ cam2_to_w

    if norm: # normalize
        T = cam2_to_cam1[:3,3]
        T /= max(1e-5, np.linalg.norm(T))

    return cam2_to_cam1.astype(np.float32)

def add_relpose(view, cam2_to_world, cam1_to_world=None):
    if cam2_to_world is not None:
        cam1_to_world = todevice(cam1_to_world, 'numpy')
        cam2_to_world = todevice(cam2_to_world, 'numpy')
        def fake_views(i):
            return [dict(camera_pose=np.eye(4) if cam1_to_world is None else cam1_to_world[i]), 
                    dict(camera_pose=cam2_to_world[i]) ]
        if cam2_to_world.ndim == 2:
            known_pose = gen_rel_pose(fake_views(slice(None)))
        else:
            known_pose = [gen_rel_pose(fake_views(i)) for i,v in enumerate(iter_views(view))]
            known_pose = torch.stack([todevice(k, view['img'].device) for k in known_pose])
        view['known_pose'] = known_pose


if __name__ == '__main__':
    # model = SPIDER_POINTMAP.from_pretrained("/cis/home/zshao14/checkpoints/aerialdust3r_relpose_dpt512_film1and2_0719/checkpoint-last.pth").to('cuda')
    # model = SPIDER_POINTMAP.from_pretrained("/cis/home/zshao14/checkpoints/aerialdust3r_relpose_dpt512_embed1and2_0725/checkpoint-final.pth").to('cuda')
    model = SPIDER_POINTMAP.from_pretrained("/cis/home/zshao14/checkpoints/aerialdust3r_relpose_dpt512_embed1and2_0728/checkpoint-final.pth").to('cuda')
    # pdb.set_trace()
    # model = SPIDER_POINTMAP.from_pretrained("/cis/home/zshao14/checkpoints/aerialdust3r_relpose_dpt512_attend1and2_0722/checkpoint-final.pth").to('cuda')
    
    thresh = 1.03
    habitat_data = Habitat_data(1000, '/cis/home/cpeng/dust3r/data/habitat_processed', split='val', resolution=512, seed=777)
    habitat_loader = torch.utils.data.DataLoader(
                habitat_data,
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
                drop_last=False)
    data = Aerial_MegaDepth_data('/cis/net/io99a/data/zshao/megadepth_aerial_data/megadepth_aerial_processed', split='val', resolution=512, seed=777)#Habitat_data(1000, '/cis/home/cpeng/dust3r/data/habitat_processed', split='val', resolution=512, seed=777)
    all_indices = list(range(len(data)))
    import random
    random.seed(777)
    sampled_indices = random.sample(all_indices, k=1000)
    sampled_data = torch.utils.data.Subset(data, sampled_indices)
    data_loader = torch.utils.data.DataLoader(
                sampled_data,
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
                drop_last=False)
    inlier_ratio = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc='AerialMega'):
            ignore_keys = set([ 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'name'])
            for view in batch:
                for name in view.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to('cuda', non_blocking=True)

            view1, view2 = batch
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with torch.cuda.amp.autocast(enabled=False):
                    cam1 = view1.get('camera_pose')
                    cam2 = view2.get('camera_pose')
                    add_relpose(view1, cam2_to_world=cam2, cam1_to_world=cam1)
                    add_relpose(view2, cam2_to_world=cam2, cam1_to_world=cam1)
                
                    pred1, _ = model(view1, view2)
                    # pred2, _ = model(view2, view1)
                    add_relpose(view1, cam2_to_world=cam1, cam1_to_world=cam2)
                    add_relpose(view2, cam2_to_world=cam1, cam1_to_world=cam2)
                    pred2, _ = model(view2, view1)

            # gt_depth = view1['depthmap']
            # cam2world = view1['camera_pose']
            # pred_pts = pred1['pts3d']
            # in_camera1 = inv(view1['camera_pose'])
            # gt_pts1 = geotrf(in_camera1, view1['pts3d'])  # B,H,W,3
            # pred_pts1 = geotrf(in_camera1, pred1['pts3d'])
            # gt_pts2 = geotrf(in_camera1, view2['pts3d'])  # B,H,W,3
            in_camera1 = inv(view1['camera_pose'])
            valid1 = view1['valid_mask'].clone()
            
            # normalize 3d points
            gt_pts1 = geotrf(in_camera1, view1['pts3d'])  # B,H,W,3
            pr_pts1 = pred1['pts3d']
            pr_pts1 = normalize_pointcloud(pr_pts1, None, 'avg_dis', valid1)
            gt_pts1 = normalize_pointcloud(gt_pts1, None, 'avg_dis', valid1)
            
            pr_z1 = pr_pts1[...,2]
            gt_z1 = gt_pts1[...,2]
            rel_11 = torch.where(pr_z1 != 0, gt_z1 / pr_z1, torch.full_like(gt_z1, thresh + 1))
            rel_12 = torch.where(gt_z1 != 0, pr_z1 / gt_z1, torch.zeros_like(pr_z1))
            max_rel1 = torch.maximum(rel_11, rel_12)
            inliers1 = ((max_rel1 > 0) & (max_rel1 < thresh)).float()

            in_camera2 = inv(view2['camera_pose'])
            valid2 = view2['valid_mask'].clone()
            
            # normalize 3d points
            gt_pts2 = geotrf(in_camera2, view2['pts3d'])  # B,H,W,3
            pr_pts2 = pred2['pts3d']
            pr_pts2 = normalize_pointcloud(pr_pts2, None, 'avg_dis', valid2)
            gt_pts2 = normalize_pointcloud(gt_pts2, None, 'avg_dis', valid2)
            pr_z2 = pr_pts2[...,2]
            gt_z2 = gt_pts2[...,2]
            rel_21 = torch.where(pr_z2 != 0, gt_z2 / pr_z2, torch.full_like(gt_z2, thresh + 1))
            rel_22 = torch.where(gt_z2 != 0, pr_z2 / gt_z2, torch.zeros_like(pr_z2))
            max_rel2 = torch.maximum(rel_21, rel_22)
            inliers2 = ((max_rel2 > 0) & (max_rel2 < thresh)).float()
            
            inlier_avg = (inliers2[valid2].mean() + inliers1[valid1].mean()) / 2
            inlier_ratio.append(inlier_avg.item())
    print(np.mean(inlier_ratio))
    pdb.set_trace()
    habitat_ratio = []
    with torch.no_grad():
        for batch in tqdm.tqdm(habitat_loader, desc='Habitat'):
            ignore_keys = set([ 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'name'])
            for view in batch:
                for name in view.keys():  # pseudo_focal
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to('cuda', non_blocking=True)

            view1, view2 = batch
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with torch.cuda.amp.autocast(enabled=False):
                    cam1 = view1.get('camera_pose')
                    cam2 = view2.get('camera_pose')
                    add_relpose(view1, cam2_to_world=cam2, cam1_to_world=cam1)
                    add_relpose(view2, cam2_to_world=cam2, cam1_to_world=cam1)
                    pred1, _ = model(view1, view2)
                    # pred2, _ = model(view2, view1)
                    add_relpose(view1, cam2_to_world=cam1, cam1_to_world=cam2)
                    add_relpose(view2, cam2_to_world=cam1, cam1_to_world=cam2)
                    pred2, _ = model(view2, view1)

            in_camera1 = inv(view1['camera_pose'])
            valid1 = view1['valid_mask'].clone()
            
            # normalize 3d points
            gt_pts1 = geotrf(in_camera1, view1['pts3d'])  # B,H,W,3
            pr_pts1 = pred1['pts3d']
            pr_pts1 = normalize_pointcloud(pr_pts1, None, 'avg_dis', valid1)
            gt_pts1 = normalize_pointcloud(gt_pts1, None, 'avg_dis', valid1)
            
            pr_z1 = pr_pts1[...,2]
            gt_z1 = gt_pts1[...,2]
            rel_11 = torch.where(pr_z1 != 0, gt_z1 / pr_z1, torch.full_like(gt_z1, thresh + 1))
            rel_12 = torch.where(gt_z1 != 0, pr_z1 / gt_z1, torch.zeros_like(pr_z1))
            max_rel1 = torch.maximum(rel_11, rel_12)
            inliers1 = ((max_rel1 > 0) & (max_rel1 < thresh)).float()

            in_camera2 = inv(view2['camera_pose'])
            valid2 = view2['valid_mask'].clone()
            
            # normalize 3d points
            gt_pts2 = geotrf(in_camera2, view2['pts3d'])  # B,H,W,3
            pr_pts2 = pred2['pts3d']
            pr_pts2 = normalize_pointcloud(pr_pts2, None, 'avg_dis', valid2)
            gt_pts2 = normalize_pointcloud(gt_pts2, None, 'avg_dis', valid2)
            pr_z2 = pr_pts2[...,2]
            gt_z2 = gt_pts2[...,2]
            rel_21 = torch.where(pr_z2 != 0, gt_z2 / pr_z2, torch.full_like(gt_z2, thresh + 1))
            rel_22 = torch.where(gt_z2 != 0, pr_z2 / gt_z2, torch.zeros_like(pr_z2))
            max_rel2 = torch.maximum(rel_21, rel_22)
            inliers2 = ((max_rel2 > 0) & (max_rel2 < thresh)).float()
            
            inlier_avg = (inliers2[valid2].mean() + inliers1[valid1].mean()) / 2
            habitat_ratio.append(inlier_avg.item())
    print(np.mean(habitat_ratio))
    pdb.set_trace()
