import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

from src.core import YAMLConfig

def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    # Create example inputs
    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])

    # Trace the model
    traced_model = torch.jit.trace(model, (data, size))

    # Save the TorchScript model
    traced_model.save(args.output_file)

    print(f'TorchScript model saved to {args.output_file}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="C:/DeepLearning/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml")
    parser.add_argument('--resume', '-r', type=str, default="C:/DeepLearning/RT-DETR/output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/best.pth")
    parser.add_argument('--output_file', '-o', type=str, default='./output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/0909_rtdetrv2_12o_xpretrain_best118.torchscript')
    
    args = parser.parse_args()

    main(args)
