import cv2
import math
import torch
import logging
from copy import deepcopy

from torch import nn
from torchvision.transforms import ToTensor

from ALIKE_code.nets.alnet_old import ALNet
from ALIKE_code.nets.soft_detect import SoftDetect
import time


class ALIKE(ALNet):
    def __init__(self,
                 # ================================== feature encoder
                 c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128, dim: int = 128,
                 agg_mode: str = 'cat',  # sum, cat, fpn
                 single_head: bool = False,
                 pe: bool = False,
                 # ================================== detect parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5,
                 n_limit: int = 0,
                 **kwargs
                 ):
        super().__init__(c1, c2, c3, c4, dim, agg_mode, single_head, pe)

        self.radius = radius
        self.top_k = top_k
        self.n_limit = n_limit
        self.scores_th = scores_th

        self.update_softdetect_parameters()

    def update_softdetect_parameters(self):
        self.softdetect = SoftDetect(radius=self.radius, top_k=self.top_k,
                                     scores_th=self.scores_th, n_limit=self.n_limit)

    def extract_dense_map(self, image, ret_dict=False):
        # ====================================================
        # check image size, should be integer multiples of 2^5
        # if it is not a integer multiples of 2^5, padding zeros
        device = image.device
        b, c, h, w = image.shape
        h_ = math.ceil(h / 8) * 8 if h % 8 != 0 else h
        w_ = math.ceil(w / 8) * 8 if w % 8 != 0 else w
        if h_ != h:
            h_padding = torch.zeros(b, c, h_ - h, w, device=device)
            image = torch.cat([image, h_padding], dim=2)
        if w_ != w:
            w_padding = torch.zeros(b, c, h_, w_ - w, device=device)
            image = torch.cat([image, w_padding], dim=3)
        # ====================================================

        scores_map, descriptor_map = super().forward(image)

        # ====================================================
        if h_ != h or w_ != w:
            descriptor_map = descriptor_map[:, :, :h, :w]
            scores_map = scores_map[:, :, :h, :w]  # Bx1xHxW
        # ====================================================

        # BxCxHxW
        descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)

        if ret_dict:
            # m = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=True)  # 8倍上采样
            # descriptor_map = m(descriptor_map) # BxCxHxW
            return {'descriptor_map': descriptor_map, 'scores_map': scores_map, }
        else:
            return descriptor_map, scores_map

    def extract(self, image):
        descriptor_map, scores_map = self.extract_dense_map(image)
        keypoints, descriptors, kptscores, scoredispersitys = self.softdetect(scores_map, descriptor_map)

        return {'keypoints': keypoints,  # B M 2
                'descriptors': descriptors,  # B M D
                'scores': kptscores,  # B M D
                'score_dispersity': scoredispersitys,
                'descriptor_map': descriptor_map,  # BxCxHxW
                'scores_map': scores_map,  # Bx1xHxW
                }

    def forward(self, img, scale_f=2 ** 0.5,
                min_scale=1., max_scale=1.,
                min_size=0., max_size=99999.,
                image_size_max=99999,
                verbose=False, n_k=0, sort=False,
                scoremap=True,
                descmap=True):
        """
        :param img: np array, HxWx3
        :param scale_f:
        :param min_scale:
        :param max_scale:
        :param min_size:
        :param max_size:
        :param verbose:
        :param n_k:
        :param sort:
        :return: keypoints, descriptors, scores
        """
        old_bm = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # speedup

        H_, W_, three = img.shape
        assert three == 3, "input image shape should be [HxWx3]"

        # ==================== image size constraint
        image = deepcopy(img)
        max_hw = max(H_, W_)
        if max_hw > image_size_max:
            ratio = float(image_size_max / max_hw)
            image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)

        # ==================== convert image to tensor
        H, W, three = image.shape
        if isinstance(image,torch.Tensor):
            image = image.unsqueeze(0)
        else:
            image = ToTensor()(image).unsqueeze(0)
        image = image.to(self.device)


        # ==================== extract keypoints at multiple scales
        start = time.time()

        s = 1.0  # current scale factor
        if verbose:
            logging.info('')
        keypoints, descriptors, scores, scores_maps, descriptor_maps = [], [], [], [], []
        while s + 0.001 >= max(min_scale, min_size / max(H, W)):
            if s - 0.001 <= min(max_scale, max_size / max(H, W)):
                nh, nw = image.shape[2:]

                # extract descriptors
                with torch.no_grad():
                    descriptor_map, scores_map = self.extract_dense_map(image)
                    keypoints_, descriptors_, scores_, _ = self.softdetect(scores_map, descriptor_map)

                if scoremap:
                    scores_maps.append(scores_map[0, 0].cpu())
                if descmap:
                    descriptor_maps.append(descriptor_map[0].cpu())
                keypoints.append(keypoints_[0])
                descriptors.append(descriptors_[0])
                scores.append(scores_[0])

                if verbose:
                    logging.info(
                        f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}. Number of keypoints {len(keypoints)}.")

            s /= scale_f

            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            image = torch.nn.functional.interpolate(image, (nh, nw), mode='bilinear', align_corners=False)

        # restore value
        torch.backends.cudnn.benchmark = old_bm

        keypoints = torch.cat(keypoints)
        descriptors = torch.cat(descriptors)
        scores = torch.cat(scores)
        keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W_ - 1, H_ - 1]])

        if sort or 0 < n_k < len(keypoints):
            indices = torch.argsort(scores, descending=True)
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]
            scores = scores[indices]

        if 0 < n_k < len(keypoints):
            keypoints = keypoints[0:n_k]
            descriptors = descriptors[0:n_k]
            scores = scores[0:n_k]

        return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores,
                'descriptor_maps': descriptor_maps,
                'scores_maps': scores_maps, 'time': time.time() - start, }


if __name__ == '__main__':
    import numpy as np
    from thop import profile

    net = ALIKE(c1=32, c2=64, c3=128, c4=128, dim=128, agg_mode='cat', res_block=True, single_head=False)

    image = np.random.random((640, 480, 3)).astype(np.float32)
    flops, params = profile(net, inputs=(image, 2 ** 0.25, 0.3, 1, 0, 9999, 9999, False, 5000, False))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))
