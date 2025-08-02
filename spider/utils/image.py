import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import h5py
import json
import pdb
import math
try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images_with_intrinsics(folder_or_list, size=512, square_ok=True, verbose=True, patch_size=16, intrinsics=None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    Ks = []
    for idx, path in enumerate(folder_content):
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if intrinsics is None:
            K = np.array([
                [1.0, 0, W1/2],
                [0, 1.0, H1/2],
                [0, 0, 1.0]
            ])
        else:
            K_ = intrinsics[idx]
            if isinstance(K_, np.ndarray):
                K = K_.copy()
            elif isinstance(K_, torch.Tensor):
                K = K_.clone()
        
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)

        W, H = img.size

        scale_x = W / W1
        scale_y = H / H1

        K[0, 0] *= scale_x  # fx
        K[0, 2] *= scale_x  # cx
        K[1, 1] *= scale_y  # fy
        K[1, 2] *= scale_y  # cy


        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
            K[0, 2] -= (cx - half) 
            K[1, 2] -= (cy - half)
        else:
            halfw = ((2 * cx) // patch_size) * patch_size / 2
            halfh = ((2 * cy) // patch_size) * patch_size / 2
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
            K[0, 2] -= (cx - halfw)
            K[1, 2] -= (cy - halfh)

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        Ks.append(K)

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, Ks


def load_images_with_intrinsics_strict(folder_or_list, size=512, square_ok=True, verbose=True, patch_size=16, intrinsics=None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    Ks = []
    for idx, path in enumerate(folder_content):
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if intrinsics is None:
            K = np.array([
                [1.0, 0, W1/2],
                [0, 1.0, H1/2],
                [0, 0, 1.0]
            ])
        else:
            K_ = intrinsics[idx]
            if isinstance(K_, np.ndarray):
                K = K_.copy()
            elif isinstance(K_, torch.Tensor):
                K = K_.clone()
        df = 16
        scale = size / max(H1, W1)
        w_new, h_new = int(round(W1*scale)), int(round(H1*scale))
        w_new = max(math.ceil(w_new // df), 1) * df
        h_new = max(math.ceil(h_new // df), 1) * df
        img = img.resize((w_new, h_new))

        W2, H2 = img.size

        scale_x = W2 / W1
        scale_y = H2 / H1

        K[0, 0] *= scale_x  # fx
        K[0, 2] *= scale_x  # cx
        K[1, 1] *= scale_y  # fy
        K[1, 2] *= scale_y  # cy

        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        Ks.append(K)

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, Ks

def load_original_images(folder_or_list, verbose=False):
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')
    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for idx, path in enumerate(folder_content):
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')       
        imgs.append(img)
        if verbose:
            print(f' - adding {path}')

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

def resize_image_with_intrinsics(imgs_ori, size=512, square_ok=True, verbose=True, patch_size=16, intrinsics=None):
    imgs = []
    Ks = []
    for idx, img in enumerate(imgs_ori):
        W1, H1 = img.size
        if intrinsics is not None:
            K_ = intrinsics[idx]
            if isinstance(K_, np.ndarray):
                K = K_.copy()
            elif isinstance(K_, torch.Tensor):
                K = K_.clone()
        else:
            K = None
        scale = size / max(H1, W1)
        w_new, h_new = int(round(W1*scale)), int(round(H1*scale))
        w_new = max(math.ceil(w_new // patch_size), 1) * patch_size
        h_new = max(math.ceil(h_new // patch_size), 1) * patch_size
        img = img.resize((w_new, h_new))

        W2, H2 = img.size

        scale_x = W2 / W1
        scale_y = H2 / H1

        if K is not None:
            K[0, 0] *= scale_x  # fx
            K[0, 2] *= scale_x  # cx
            K[1, 1] *= scale_y  # fy
            K[1, 2] *= scale_y  # cy
            Ks.append(K)

        if verbose:
            print(f' - adding with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=torch.from_numpy(np.int32(
            [img.size[::-1]])), idx=len(imgs), instance=str(len(imgs))))
       
    assert imgs, 'no images'
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, Ks

def load_two_images_with_H(img1_path, img2_path, size=512, square_ok=True, verbose=True, patch_size=16, H_ori=None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """

    root = ''
    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    
    ### img 1
    if img1_path.lower().endswith(supported_images_extensions):
        img = exif_transpose(PIL.Image.open(os.path.join(root, img1_path))).convert('RGB')
        W1, H1 = img.size
        img = _resize_pil_image(img, size)

        W, H = img.size
        scale_x = W / W1
        scale_y = H / H1

        T_scale1 = np.array([
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1]
                ])

        cx, cy = W//2, H//2
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        T_crop1 = np.array([
                    [1, 0, -(cx - halfw)],
                    [0, 1, -(cy - halfh)],
                    [0, 0, 1.0]
                ])
        T1 = T_crop1 @ T_scale1

        W2, H2 = img.size
        if verbose:
            print(f' - adding {img1_path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
    

         ### img 2
    if img2_path.lower().endswith(supported_images_extensions):
        img = exif_transpose(PIL.Image.open(os.path.join(root, img2_path))).convert('RGB')
        W1, H1 = img.size
        img = _resize_pil_image(img, size)

        W, H = img.size
        scale_x = W / W1
        scale_y = H / H1

        T_scale2 = np.array([
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1.0]
                ])

        cx, cy = W//2, H//2
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        T_crop2 = np.array([
                    [1, 0, -(cx - halfw)],
                    [0, 1, -(cy - halfh)],
                    [0, 0, 1.0]
                ])
        T2 = T_crop2 @ T_scale2
        W2, H2 = img.size
        if verbose:
            print(f' - adding {img2_path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
    
    if H_ori is not None:
        # pdb.set_trace()
        H_new = T2 @ H_ori @ np.linalg.inv(T1)
    else:
        H_new = None

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, H_new

def load_two_images_with_H_strict(img1_path, img2_path, size=512, square_ok=True, verbose=True, patch_size=16, H_ori=None):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """

    root = ''
    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.ppm']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    
    ### img 1
    if img1_path.lower().endswith(supported_images_extensions):
        img = exif_transpose(PIL.Image.open(os.path.join(root, img1_path))).convert('RGB')
        W1, H1 = img.size
        df = 16
        scale = size / max(H1, W1)
        w_new, h_new = int(round(W1*scale)), int(round(H1*scale))
        w_new = max((w_new // df), 1) * df
        h_new = max((h_new // df), 1) * df
        img = img.resize((w_new, h_new))

        W2, H2 = img.size

        scale_x = W2 / W1
        scale_y = H2 / H1

        T1 = np.array([
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1]
                ])
        if verbose:
            print(f' - adding {img1_path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
    

         ### img 2
    if img2_path.lower().endswith(supported_images_extensions):
        img = exif_transpose(PIL.Image.open(os.path.join(root, img2_path))).convert('RGB')
        W1, H1 = img.size
        df = 16
        scale = size / max(H1, W1)
        w_new, h_new = int(round(W1*scale)), int(round(H1*scale))
        w_new = max((w_new // df), 1) * df
        h_new = max((h_new // df), 1) * df
        img = img.resize((w_new, h_new))

        W2, H2 = img.size

        scale_x = W2 / W1
        scale_y = H2 / H1

        T2 = np.array([
                    [scale_x, 0, 0],
                    [0, scale_y, 0],
                    [0, 0, 1.0]
                ])

        if verbose:
            print(f' - adding {img2_path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
    
    if H_ori is not None:
        # pdb.set_trace()
        H_new = T2 @ H_ori @ np.linalg.inv(T1)
    else:
        H_new = None

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, H_new

