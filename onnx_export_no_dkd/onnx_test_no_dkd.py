###双输入图片测试
import math
import time
from copy import deepcopy

import onnxruntime as ort
import cv2
import numpy as np
import torch
from PIL import Image
import onnx
from onnx_export.onnxmodel import ONNXModel
from soft_detect import DKD


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check_model(onnx_path):
    # 测试转换模型是否有问题
    onnx_model = onnx.load(onnx_path)
    check = onnx.checker.check_model(onnx_model)
    print('check', check) # check 为 none 则正常

def mnn_mather(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.75] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()


def plot_keypoints(image, kpts, radius=2, color=(0, 0, 255)):
    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        x0, y0 = kpt
        cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out


def plot_matches(image0,image1,kpts0,kpts1,matches,radius=2,color=(255, 0, 0)):
    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1

    mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)

    points_out = out.copy()
    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1
        mcolor = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        cv2.line(out, (x0, y0), (x1 + W0, y1),
                 color=mcolor,
                 thickness=1,
                 lineType=cv2.LINE_AA)

    cv2.putText(out, str(len(mkpts0)),
                (out.shape[1] - 150, out.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    return out,points_out


def post_deal(flg,W,H,scores_map, descriptor_map,radius=2,top_k=2000, scores_th=0.2,n_limit=5000,sort=False):
    descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)
    # if flg:
    #     descriptor_map = descriptor_map[:, :, :H, :W]
    #     scores_map = scores_map[:, :, :H, :W]  # Bx1xHxW

    keypoints, descriptors, scores, _ = DKD(radius=radius, top_k=top_k,scores_th=scores_th, n_limit=n_limit).forward(scores_map, descriptor_map)
    keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
    keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])
    if sort:
        indices = torch.argsort(scores, descending=True)
        keypoints = keypoints[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

    return {'keypoints': keypoints.cpu().numpy(),
            'descriptors': descriptors.cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'scores_map': scores_map.cpu().numpy(),}

# def pre_deal_tensor(img,flg=False):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     image = torch.from_numpy(img_rgb).to(torch.float32).permute(2, 0, 1)[None] / 255.0
#     b, c, h, w = image.shape
#     h_ = math.ceil(h / 32) * 32 if h % 32 != 0 else h
#     w_ = math.ceil(w / 32) * 32 if w % 32 != 0 else w
#     if h_ != h:
#         h_padding = torch.zeros(b, c, h_ - h, w)
#         image = torch.cat([image, h_padding], dim=2)
#     if w_ != w:
#         w_padding = torch.zeros(b, c, h_, w_ - w)
#         image = torch.cat([image, w_padding], dim=3)
#     if h_ != h or w_ != w:
#         flg = True
#     return image,flg,h,w

def pre_deal_np(img,flg=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img_rgb.transpose(2, 0, 1)[None] / 255.0
    b, c, h, w = image.shape
    h_ = math.ceil(h / 8) * 8 if h % 8 != 0 else h
    w_ = math.ceil(w / 8) * 8 if w % 8 != 0 else w
    if h_ > h:
        h_padding = np.zeros((b, c, h_ - h, w))
        image = np.concatenate([image, h_padding], dtype=np.float32,axis=2)
    if w_ > w:
        w_padding = np.zeros((b, c, h_, w_ - w))
        image = np.concatenate([image, w_padding], dtype=np.float32,axis=3)
    if h_ != h or w_ != w:
        flg = True
    image = image.astype(np.float32)
    return image,flg,h,w

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main(model_file):
    # 读取图片
    path1 = '/media/xin/work1/github_pro/ALIKE/test_img/parker/L/1614045104206922_L.png'
    path2 = '/media/xin/work1/github_pro/ALIKE/test_img/parker/R/1614045104206922_R.png'
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img_rgb1,flg1,H1,W1 = pre_deal_np(img1)
    img_rgb2,flg2,H2,W2 = pre_deal_np(img2)
    # 加载 onnx_export
    start = time.time()
    model = ONNXModel(model_file)
    print("时间：",time.time()-start)
    scores_map1,descriptor_map1 = model.forward(img_rgb1)
    descriptor_map1 = torch.from_numpy(descriptor_map1)
    scores_map1 = torch.from_numpy(scores_map1)
    output1 = post_deal(flg1,W1,H1,scores_map1, descriptor_map1)
    scores_map2,descriptor_map2 = model.forward(img_rgb2)
    descriptor_map2  = torch.from_numpy(descriptor_map2)
    scores_map2 = torch.from_numpy(scores_map2)
    output2 = post_deal(flg2,W2,H2,scores_map2, descriptor_map2)
    kpts = output1['keypoints']
    desc = output1['descriptors']
    kpts_ref = output2['keypoints']
    desc_ref = output2['descriptors']
    matches = mnn_mather(desc, desc_ref)
    vis_img, points_out = plot_matches(img1, img2, kpts, kpts_ref, matches)
    cv2.putText(points_out, str(len(kpts)),
                (points_out.shape[1] - 150, points_out.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow('points', points_out)
    cv2.imshow('matches', vis_img)
    cv2.imwrite('./res/point.png',points_out)
    cv2.imwrite('./res/desc.png',vis_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print('success!')


if __name__ == '__main__':
    model_file = '/media/xin/work1/github_pro/ALIKE/onnx_export_no_dkd/alnet.onnx'
    check_model(model_file)
    main(model_file)
