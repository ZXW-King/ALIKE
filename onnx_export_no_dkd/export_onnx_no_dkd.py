"""
pth转onnx
"""
import argparse
import math
import numpy as np
import torch

from onnx_export.alike_onnx import ALike, configs
from alnet import ALNet
# from onnx_export_no_dkd.alnet_x import ALNet
def GetArgs():
    parser = argparse.ArgumentParser(description='ALIKE model export demo.')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l'], default="alike-n",help="The model configuration")
    parser.add_argument('--model_path', default="", help="The model path, The default is open source model")
    parser.add_argument('--export_onnx_path', type=str, default='', help='model save path.')
    parser.add_argument('--device', type=str, default='cpu', help="Running device (default: cpu).")
    args = parser.parse_args()
    return args

def main():
    args = GetArgs()
    w, h = 640, 400
    model = ALNet(**configs[args.model])
    device = args.device
    # load model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    onnx_input = torch.rand(1, 3, h, w).to(device)

    torch.onnx.export(
        model,
        onnx_input,
        args.export_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output1','output2']
    )
    print("导出模型成功！")

if __name__ == '__main__':
    main()
