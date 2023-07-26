"""
ckpt转pth转onnx
"""
import argparse
import torch

from onnx_export_no_dkd.alnet import ALNet
def GetArgs():
    parser = argparse.ArgumentParser(description='ALIKE model export demo.')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l'], default="alike-n",help="The model configuration")
    parser.add_argument('--model_path', default="", help="The model path, The default is open source model")
    parser.add_argument('--export_onnx_path', type=str, default='alnet_new.onnx', help='model save path.')
    args = parser.parse_args()
    return args

def get_ckpt_model(checkpoint_path,device):
    model = ALNet(c1=16, c2=32, c3=64, c4=128, dim=128, single_head=True)
    model.load_from_checkpoint(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model

def ckpt2pth(checkpoint_path,device):
    # 加载模型
    model = get_ckpt_model(checkpoint_path, device)
    # 直接将模型保存为pth
    torch.save(model.state_dict(),'alnet.pth')


def pth2onnx(model,model_path,export_onnx_path):
    configs = {
        'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, },
        'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, },
        'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, },
        'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, },
    }
    device = 'cpu'
    w, h = 640, 400
    input = torch.rand(1, 3, h, w).to(device)

    if model_path.endswith('.ckpt'):
        ckpt2pth(model_path,device)
        model_path = 'alnet.pth'
    elif model_path.endswith('.pth'):
        pass
    else:
        raise "模型格式错误"

    model = ALNet(**configs[model])
    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    torch.onnx.export(
        model,
        input,
        export_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output1','output2']
    )
    print("导出模型成功！")

if __name__ == '__main__':
    args = GetArgs()
    pth2onnx(args.model,args.model_path,args.export_onnx_path)
