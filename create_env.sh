conda create -n torch1.10 python=3.7
conda activate torch1.10
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge opencv
pip install -r requirements.txt
cd libs/torchlights
pip install -e .