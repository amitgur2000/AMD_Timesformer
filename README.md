# AMD_Timesformer
Timesformer on OCT scan to classify AMD stages 


# To install

git clone this repo

conda deactivate 

conda env create -n timesformer --file timesformer.yaml

conda activate timesformer

cd AMD_Timesformer

torchvision: pip install torchvision or conda install torchvision -c pytorch

fvcore pip (might have issues): pip install 'git+https://github.com/facebookresearch/fvcore'

fvcore conda: conda install -y -c conda-forge fvcore

iopath: conda install -y -c conda-forge iopath

simplejson: pip install simplejson

einops: pip install einops

timm: pip install timm   # conda install -c conda-forge timm

PyAV: conda install -y -c conda-forge av

psutil: pip install psutil  # conda install -c conda-forge psutil

scikit-learn: pip install scikit-learn  # conda install -y -c anaconda scikit-learn

conda install -y -c conda-forge matplotlib

conda install -y -c conda-forge tensorboard

conda install -y -c conda-forge simplejson

python -m pip install opencv-python    # Trying to install through conda didn't work: conda install -c conda-forge opencv

conda install -y -c conda-forge einops

For CUDA 11.6 and above, install: conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

For CUDA 11.8 or CUDA 12:  conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia



cd AMD_Timesformer