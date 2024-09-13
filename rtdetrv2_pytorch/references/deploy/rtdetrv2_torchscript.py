import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw


def draw(images, labels, boxes, scores, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        category_name = [ "white", "white_front", "blue_s", "blue_s_pin", "blue_m", "blue_m_pin", "blue_l", "blue_l_pin", "locker_front", "locker_side"]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{category_name[i]} {round(scrs[j].item(), 2)}", fill='blue')
            # draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill='blue')

        im.save(f'results_{i}.jpg')

def main(args):
    """main"""
    # Load the TorchScript model
    model = torch.jit.load(args.model_path)
    model.to(args.device)
    model.eval()

    # Load and preprocess the image
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    # Perform inference
    with torch.no_grad():
        output = model(im_data, orig_size)
    
    # Extract outputs
    labels, boxes, scores = output

    # Draw results on the image
    draw([im_pil], labels, boxes, scores)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default = "C:/DeepLearning/RT-DETR/output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/0909_rtdetrv2_12o_xpretrain_best118.torchscript", help='Path to the TorchScript model')
    parser.add_argument('--im-file', '-f', type=str, default = "C:/PE_data/pe_data_0711-0812/test/images/153.png", help='Path to the input image')
    parser.add_argument('--output_image_path', '-o', type=str, default = "output/rtdetrv2_r18vd_120e_coco/sep09/no_pretrain/results", help='Path to save the output image')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run inference on')

    args = parser.parse_args()
    main(args)
