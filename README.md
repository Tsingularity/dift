# Diffusion Features (DIFT)
This repository contains code for paper "Emergent Correspondence from Image Diffusion". 

### [Project Page](https://diffusionfeatures.github.io/) | [Paper](https://arxiv.org/abs/2306.03881) | [Colab Demo](https://colab.research.google.com/drive/1tUTJ3UJxbqnfvUMvYH5lxcqt0UdUjdq6?usp=sharing)

![video](./assets/teaser.gif)

## Prerequisites
If you have a Linux machine, you could either set up the python environment using the following command:
```
conda env create -f environment.yml
conda activate dift
```
or create a new conda environment and install the packages manually using the 
shell commands in [setup_env.sh](setup_env.sh).

## Interactive Demo: Give it a Try!
We provide an interative jupyter notebook [demo.ipynb](demo.ipynb) to demonstrate the semantic correspondence established by DIFT, and you could try on your own images! After loading two images, you could left-click on an interesting point of the source image on the left, then after 1 or 2 seconds, the corresponding point on the target image will be displayed as a red point on the right, together with a heatmap showing the per-pixel cosine distance calculated using DIFT. Here're two examples on cat and guitar:

<table>
    <tr>
        <td><img src="./assets/demo_cat_480p.gif" alt="demo cat" style="width:100%;"></td>
        <td><img src="./assets/demo_guitar_480p.gif" alt="demo cat" style="width:100%;"></td>
    </tr>
</table>

If you don't have a local GPU, you can also use the provided [Colab Demo](https://colab.research.google.com/drive/1tUTJ3UJxbqnfvUMvYH5lxcqt0UdUjdq6?usp=sharing).

## Extract DIFT for a given image
You could use the following [command](extract_dift.sh) to extract DIFT from a given image, and save it as a torch tensor. These arguments are set to the same as in the semantic correspondence tasks by default.
```
python extract_dift.py \
    --input_path ./assets/cat.png \
    --output_path dift_cat.pt \
    --img_size 768 768 \
    --t 261 \
    --up_ft_index 1 \
    --prompt 'a photo of a cat' \
    --ensemble_size 8
```
Here're the explanation for each argument:
- `input_path`: path to the input image file.
- `output_path`: path to save the output features as torch tensor.
- `img_size`: the width and height of the resized image before fed into diffusion model. If set to 0, then no resize operation would be performed thus it will stick to the original image size. It is set to [768, 768] by default. You can decrease this if encountering memory issue.
- `t`: time step for diffusion, choose from range [0, 1000], must be an integer. `t=261` by default for semantic correspondence.
- `up_ft_index`: the index of the U-Net upsampling block to extract the feature map, choose from [0, 1, 2, 3]. `up_ft_index=1` by default for semantic correspondence.
- `prompt`: the prompt used in the diffusion model.
- `ensemble_size`: the number of repeated images in each batch used to get features. `ensemble_size=8` by default. You can reduce this value if encountering memory issue.

The output DIFT tensor spatial size is determined by both `img_size` and `up_ft_index`. If `up_ft_index=0`, the output size would be 1/32 of `img_size`; if `up_ft_index=1`, it would be 1/16; if `up_ft_index=2 or 3`, it would be 1/8. 

## Application: Edit Propagation
Using DIFT, we can propagate edits in one image to others that share semantic correspondences, even cross categories and domains:
<img src="./assets/edit_cat.gif" alt="edit cat" style="width:90%;">

Check out more videos and visualizations in the [project page](https://diffusionfeatures.github.io/). 
