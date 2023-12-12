import torch
import sys
import os
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import skimage.io as io
import torch.nn.functional as F
from src.models.dift_sd import SDFeaturizer4Eval
from src.models.dift_adm import ADMFeaturizer4Eval

class HPatchDataset(Dataset):
    def __init__(self, imdir, spdir):
        self.imfs = []
        for f in os.listdir(imdir):
            scene_dir = os.path.join(imdir, f)
            self.imfs.extend([os.path.join(scene_dir, '{}.ppm').format(ind) for ind in range(1, 7)])
        self.spdir = spdir

    def __getitem__(self, item):
        imf = self.imfs[item]
        im = io.imread(imf)
        name, idx = imf.split('/')[-2:]
        coord = np.loadtxt(os.path.join(self.spdir, f'{name}-{idx[0]}.kp')).astype(np.float32)
        out = {'coord': coord, 'imf': imf}
        return out

    def __len__(self):
        return len(self.imfs)


def main(args):
    for arg in vars(args):
        value = getattr(args,arg)
        if value is not None:
            print('%s: %s' % (str(arg),str(value)))

    dataset = HPatchDataset(imdir=args.hpatches_path, spdir=args.kpts_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if args.dift_model == 'sd':
        dift = SDFeaturizer4Eval()
    elif args.dift_model == 'adm':
        dift = ADMFeaturizer4Eval()

    with torch.no_grad():
        for data in tqdm(data_loader):
            img_path = data['imf'][0]
            img = Image.open(img_path)
            w, h = img.size
            coord = data['coord'].to('cuda')
            c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord.device).float()
            coord_norm = (coord - c) / c

            feat = dift.forward(img,
                                img_size=args.img_size,
                                t=args.t,
                                up_ft_index=args.up_ft_index,
                                ensemble_size=args.ensemble_size)

            feat = F.grid_sample(feat, coord_norm.unsqueeze(2)).squeeze(-1)
            feat = feat.transpose(1, 2)

            desc = feat.squeeze(0).detach().cpu().numpy()
            kpt = coord.cpu().numpy().squeeze(0)

            out_dir = os.path.join(args.save_path, os.path.basename(os.path.dirname(img_path)))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'{os.path.basename(img_path)}.dift_{args.dift_model}'), 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=kpt,
                    scores=[],
                    descriptors=desc
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--hpatches_path', type=str, default='/scratch/dift_release/d2-net/hpatches_sequences/hpatches-sequences-release', help='path to hpatches dataset')
    parser.add_argument('--kpts_path', type=str, default='./superpoint-1k', help='path to 1k superpoint keypoints')
    parser.add_argument('--save_path', type=str, default='./hpatches_results', help='path to save features')
    parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--t', default=261, type=int, help='t for diffusion')
    parser.add_argument('--up_ft_index', default=1, type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--ensemble_size', default=8, type=int, help='ensemble size for getting an image ft map')
    args = parser.parse_args()
    main(args)