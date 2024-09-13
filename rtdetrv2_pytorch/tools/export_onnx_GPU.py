import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 

from src.core import YAMLConfig

def main(args, ):
    """main"""
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
        print('Not loading model.state_dict, using default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    # Move the model to CUDA
    model = Model().cuda()

    # Move the data to CUDA
    data = torch.rand(1, 3, 640, 640).cuda()  # Random tensor for example
    size = torch.tensor([[640, 640]]).cuda()  # Random size for example

    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    # Export to ONNX with CUDA-enabled model
    torch.onnx.export(
        model, 
        (data, size), 
        args.output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export ONNX model done...')

    # if args.simplify:
    #     import onnx 
    #     import onnxsim
    #     dynamic = True 
    #     input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
    #     onnx_model_simplify, check = onnxsim.simplify(args.file_name, input_shapes=input_shapes, dynamic_input_shape=dynamic)
    #     onnx.save(onnx_model_simplify, args.file_name)
    #     print(f'Simplify ONNX model {check}...')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="C:/DeepLearning/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml")
    parser.add_argument('--resume', '-r', type=str, default="C:/DeepLearning/RT-DETR/output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/best.pth")
    parser.add_argument('--output_file', '-o', type=str, default='./output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/0909_rtdetrv2_12o_xpretrain_best118_gpu.onnx')
    parser.add_argument('--check', action='store_true', default=True,)
    parser.add_argument('--simplify', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
