import argparse
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer

def main(args):
    dift = SDFeaturizer(args.model_id)
    img = Image.open(args.input_path).convert('RGB')
    if args.img_size[0] > 0:
        img = img.resize(args.img_size)
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    ft = dift.forward(img_tensor,
                      prompt=args.prompt,
                      t=args.t,
                      up_ft_index=args.up_ft_index,
                      ensemble_size=args.ensemble_size)
    ft = torch.save(ft.squeeze(0).cpu(), args.output_path) # save feature in the shape of [c, h, w]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''extract dift from input image, and save it as torch tenosr,
                    in the shape of [c, h, w].''')
    
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1', type=str, 
                        help='model_id of the diffusion model in huggingface')
    parser.add_argument('--t', default=261, type=int, 
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=1, type=int, choices=[0, 1, 2 ,3],
                        help='which upsampling block of U-Net to extract the feature map')
    parser.add_argument('--prompt', default='', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument('--ensemble_size', default=8, type=int, 
                        help='number of repeated images in each batch used to get features')
    parser.add_argument('--input_path', type=str,
                        help='path to the input image file')
    parser.add_argument('--output_path', type=str, default='dift.pt',
                        help='path to save the output features as torch tensor')
    args = parser.parse_args()
    main(args)