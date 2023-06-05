conda create -n dift python=3.10
conda activate dift

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install xformers -c xformers
pip install jupyterlab
pip install diffusers[torch]==0.15.0
pip install -U matplotlib
pip install transformers
pip install ipympl
pip install triton