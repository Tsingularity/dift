import numpy as np
import argparse
import os
import torch
from tqdm import tqdm
import cv2
import torch.nn.functional as F

def mnn_matcher(descriptors_a, descriptors_b, metric='cosine'):
    device = descriptors_a.device
    if metric == 'cosine':
        descriptors_a = F.normalize(descriptors_a)
        descriptors_b = F.normalize(descriptors_b)
        sim = descriptors_a @ descriptors_b.t()
    elif metric == 'l2':
        dist = torch.sum(descriptors_a**2, dim=1, keepdim=True) + torch.sum(descriptors_b**2, dim=1, keepdim=True).t() - \
           2 * descriptors_a.mm(descriptors_b.t())
        sim = -dist
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

def generate_read_function(save_path, method, extension='ppm', top_k=None):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(save_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            if len(aux['scores']) != 0:
                ids = np.argsort(aux['scores'])[-top_k :]
                if len(aux['scores'].shape) == 2:
                    scores = aux['scores'][0]
                elif len(aux['scores'].shape) == 1:
                    scores = aux['scores']
                ids = np.argsort(scores)[-top_k :]
                return aux['keypoints'][ids, :], aux['descriptors'][ids, :]
            else:
                return aux['keypoints'][:, :2], aux['descriptors']
    return read_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--hpatches_path', type=str, default='/scratch/dift_release/d2-net/hpatches_sequences/hpatches-sequences-release', help='path to hpatches dataset')
    parser.add_argument('--save_path', type=str, default='./hpatches_results', help='path to save features')
    parser.add_argument('--feat', choices=['dift_sd', 'dift_adm'], default='dift_sd', help="which feature to use")
    parser.add_argument('--metric', choices=['cosine', 'l2'], default='cosine', help="which distance metric to use")
    parser.add_argument('--mode', choices=['ransac', 'lmeds'], default='lmeds', help="which method to use when calculating homography")
    args = parser.parse_args()

    seq_names = sorted(os.listdir(args.hpatches_path))
    read_function = generate_read_function(args.save_path, args.feat)
    th = np.linspace(1, 5, 3)

    i_accuracy = []
    v_accuracy = []

    for seq_idx, seq_name in tqdm(enumerate(seq_names)):
        keypoints_a, descriptors_a = read_function(seq_name, 1)
        keypoints_a, unique_idx = np.unique(keypoints_a, return_index=True, axis=0)
        descriptors_a = descriptors_a[unique_idx]

        h, w = cv2.imread(os.path.join(args.hpatches_path, seq_name, '1.ppm')).shape[:2]

        for im_idx in range(2, 7):
            h2, w2 = cv2.imread(os.path.join(args.hpatches_path, seq_name, '{}.ppm'.format(im_idx))).shape[:2]
            keypoints_b, descriptors_b = read_function(seq_name, im_idx)
            keypoints_b, unique_idx = np.unique(keypoints_b, return_index=True, axis=0)
            descriptors_b = descriptors_b[unique_idx]

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).cuda(),
                torch.from_numpy(descriptors_b).cuda(),
                metric=args.metric
            )

            H_gt = np.loadtxt(os.path.join(args.hpatches_path, seq_name, "H_1_" + str(im_idx)))
            pts_a = keypoints_a[matches[:, 0]].reshape(-1, 1, 2).astype(np.float32)
            pts_b = keypoints_b[matches[:, 1]].reshape(-1, 1, 2).astype(np.float32)

            if args.mode == 'ransac':
                H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=3)
            elif args.mode == 'lmeds':
                H, mask = cv2.findHomography(pts_a, pts_b, cv2.LMEDS, ransacReprojThreshold=3)

            corners = np.array([[0, 0, 1],
                                [0, h-1, 1],
                                [w - 1, 0, 1],
                                [w - 1, h - 1, 1]])

            real_warped_corners = np.dot(corners, np.transpose(H_gt))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            warped_corners = np.dot(corners, np.transpose(H))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            correctness = mean_dist <= th

            if seq_name[0] == 'i':
                i_accuracy.append(correctness)
            elif seq_name[0] == 'v':
                v_accuracy.append(correctness)

    i_accuracy = np.array(i_accuracy)
    v_accuracy = np.array(v_accuracy)
    i_mean_accuracy = np.mean(i_accuracy, axis=0)
    v_mean_accuracy = np.mean(v_accuracy, axis=0)
    overall_mean_accuracy = np.mean(np.concatenate((i_accuracy, v_accuracy), axis=0), axis=0)
    print('overall_acc: {}, i_acc: {}, v_acc: {}'.format(
        overall_mean_accuracy * 100, i_mean_accuracy * 100, v_mean_accuracy * 100))