import torch

class DINOFeaturizer:
    def __init__(self, dino_id='dino_vitb8'):
        self.model = torch.hub.load('facebookresearch/dino:main', dino_id).eval().cuda()

    @torch.no_grad()
    def forward(self, img_tensor, block_index):
        h = img_tensor.shape[2] // 8
        w = img_tensor.shape[3] // 8
        n = 12 - block_index
        out = self.model.get_intermediate_layers(img_tensor, n=n)[0][0, 1:, :] # hw, c
        dim = out.shape[1]
        out = out.transpose(0, 1).view(1, dim, h, w)
        return out