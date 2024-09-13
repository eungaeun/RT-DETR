"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision.transforms as T

import numpy as np 
import onnxruntime as ort 
from PIL import Image, ImageDraw


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        category_name = [ "white", "white_front", "blue_s", "blue_s_pin", "blue_m", "blue_m_pin", "blue_l", "blue_l_pin", "locker_front", "locker_side"]
        print('lab type: ', lab.dtype)
        print('scrtype: ', scr.dtype)
        print('box type: ', box.dtype)
        print('scr: ', scr)

        for b in box:
            draw.rectangle(list(b), outline='red',)
            # draw.text((b[0], b[1]), text=str(lab[i].item()), fill='blue', )
            draw.text((b[0], b[1]), text=f"{category_name[i]} {round(scr[i].item(), 2)}", fill='blue', size = 10)

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    sess = ort.InferenceSession(args.onnx_file)
    print(ort.get_device())

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None]

    output = sess.run(
        # output_names=['labels', 'boxes', 'scores'],
        output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
    )

    labels, boxes, scores = output
    print("ori type: ", orig_size.dtype)
    # print('scores: ', scores)
    # print(boxes)

    draw([im_pil], labels, boxes, scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-file', type=str,default = "C:/DeepLearning/RT-DETR/output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/0909_rtdetrv2_12o_xpretrain_best118.onnx" )
    parser.add_argument('--im-file', type=str, default = "C:/PE_data/pe_data_mini/test/images/0812c1_001.png")
    # parser.add_argument('--im-file', type=str, default = "C:/PE_data/pe_data_0711-0812/test/images/153.png")
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)

