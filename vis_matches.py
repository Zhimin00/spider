from spider.model import SPIDER
from spider.utils.utils import compute_relative_pose, estimate_pose, compute_pose_error, pose_auc, match_symmetric, sample_symmetric, to_pixel_coordinates
from spider.inference import inference
import spider.utils.path_to_dust3r #noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
import pdb
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
import torch.nn.functional as F
import os

def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False):
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)

if __name__ == '__main__':
    device = 'cuda'
    model = SPIDER.from_pretrained("/cis/home/zshao14/checkpoints/spider_warp/checkpoint-last.pth").to(device)
    # im1_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/A01/A01_s07/input/images/image_000001.jpg'
    # im2_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/A01/A01_s07/input/images/image_000004.jpg'
    # im1_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000075.JPG'
    # im2_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/BLH0001/input/images/image_000003.JPG'
    # im1_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/M07/airborne/images/image_000003.jpg'
    # im2_path = '/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/M07/ground/images/image_000001.jpg'

    dir_name = 'SaintPeter'#'habitat_vangogh'#'co3d_apple'# 'SaintPeter'#'toronto'
    im1_path = f'/cis/home/zshao14/Downloads/spider/assets/{dir_name}/{dir_name}_A.jpg'
    im2_path = f'/cis/home/zshao14/Downloads/spider/assets/{dir_name}/{dir_name}_B.jpg'
    save_dir = f'/cis/home/zshao14/Downloads/spider/assets/{dir_name}'
    
    imgs = load_images([im1_path, im2_path], size=512, square_ok=True)
    image_pairs = []
    image_pairs.append((imgs[0], imgs[1]))
    image_pairs.append((imgs[1], imgs[0]))
    res = inference(image_pairs, model, device, batch_size=1, verbose=True)
    view1, view2 = res['view1'], res['view2']
    warp1, certainty1, warp2, certainty2 = match_symmetric(res['corresps'])
    sparse_matches, _ = sample_symmetric(warp1, certainty1, warp2, certainty2, num=5000)
    
    # H0, W0 = imgs[0]['true_shape'][0]
    # H1, W1 = imgs[1]['true_shape'][0]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'][0] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    kpts1, kpts2 = to_pixel_coordinates(sparse_matches, H0, W0, H1, W1)
    matches_im0, matches_im1 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
    
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]


    num_matches = len(matches_im0)
    n_viz = 100
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    # for i in range(num_matches):
    #     (x0, y0), (x1, y1) = matches_im0[i].T, matches_im1[i].T
    #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (num_matches - 1)), scalex=False, scaley=False)
    # pl.show(block=True)


    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # pl.show(block=True)
    pl.savefig(os.path.join(save_dir, 'matches.png'), dpi=300, bbox_inches='tight')
    pl.close()
    im2_transfer_rgb = F.grid_sample(
        view2['img'][0][None], warp1[:,:, 2:][None], mode="bilinear", align_corners=False
        )[0] ###H1, W1
    im1_transfer_rgb = F.grid_sample(
        view1['img'][0][None], warp2[:, :, :2][None], mode="bilinear", align_corners=False
        )[0] ###H2, W2

    white_im1 = torch.ones((H0,W0))
    white_im2 = torch.ones((H1,W1))
    vis_im1 = certainty1 * im2_transfer_rgb + (1 - certainty1) * white_im1
    vis_im2 = certainty2 * im1_transfer_rgb + (1 - certainty2) * white_im2
    tensor_to_pil(vis_im1, unnormalize=False).save(os.path.join(save_dir, f'warp_im1.jpg'))
    tensor_to_pil(vis_im2, unnormalize=False).save(os.path.join(save_dir, f'warp_im2.jpg'))
    # # visualize a few matches
    # import numpy as np
    # import torch
    # import torchvision.transforms.functional
    # from matplotlib import pyplot as pl
    # import cv2
    # from PIL import Image
    # W_A, H_A = Image.open(im1_path).size
    # W_B, H_B = Image.open(im2_path).size
    # img1, img2 = cv2.imread(im1_path), cv2.imread(im2_path)

    # canvas = np.zeros((max(H_A, H_B), W_A + W_B, 3), dtype=np.uint8)
    # canvas[:H_A, :W_A] = img1
    # canvas[:H_B, W_A:] = img2
    # offset = np.array([W_A, 0])

    # H0, W0, H1, W1 = *view1['img'].shape[-2:], *view2['img'].shape[-2:]
        
    # import pdb
    # pdb.set_trace()
    # # Draw matches
    # import cv2
    # import numpy as np
    # import matplotlib.pyplot as plt
    # num_matches = len(matches_im0)

    # colors = plt.cm.get_cmap('hsv', num_matches)

    # for i in range(num_matches):
    #     pt1 = pts0[i].astype('int32')  # Keypoint in img1
    #     pt2 = pts1[i].astype('int32') + offset  # Keypoint in img2 with offset

    #     color = colors(i)[:3]
    #     color = tuple(int(c * 255) for c in color[::-1])

    #     cv2.circle(canvas, tuple(pt1), 2, color, -1)
    #     cv2.circle(canvas, tuple(pt2), 2, color, -1)
    #     cv2.line(canvas, tuple(pt1), tuple(pt2), color, 1)


    # # for i in range(len(matches_im0)):
    # #     pt1 = pts0[i].astype('int32')  # Keypoint in img1
    # #     pt2 = pts1[i].astype('int32') + offset  # Keypoint in img2 with offset
    # #     # Draw circles at the keypoints
    # #     cv2.circle(canvas, tuple(pt1), 2, (0, 0, 255), -1)  # Red for img1
    # #     cv2.circle(canvas, tuple(pt2), 2, (0, 0, 255), -1)  # Red for img2
    # #     # Draw line connecting the keypoints
    # #     cv2.line(canvas, tuple(pt1), tuple(pt2), (0, 255, 0), 2)  # Green line

    # cv2.imwrite('/cis/net/r24a/data/zshao/data/wriva_processed_data/cross-view/vis_all2/M07/aerialmast3r2.png', canvas)
    # print(f"Finished: mast3r")

    # H, W = model.get_output_resolution()
    #     print(H,W)
    #     im1 = Image.open(im1_path).resize((H, W))
    #     im2 = Image.open(im2_path).resize((H, W))
    #     x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    #     x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
    #     im2_transfer_rgb = F.grid_sample(
    #     x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    #     )[0]
    #     im1_transfer_rgb = F.grid_sample(
    #     x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    #     )[0]
    #     warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    #     white_im = torch.ones((H,2*W),device=device)
    #     vis_im = certainty * warp_im + (1 - certainty) * white_im
    #     tensor_to_pil(vis_im, unnormalize=False).save(os.path.join(output_path, f'{name}_warp.jpg'))
    #     print(f"Finished: {name}_warp")