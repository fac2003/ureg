# Installation

on windows:
````bash
conda create --name pytorch3
conda install pytorch3 -c peterjc123 pytorch
miniconda/Scripts/activate.bat pytorch3
````
Use the pip.exe in miniconda for the following.

on mac:
````bash
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
````
Common to all platforms:
````bash
pip install torchvision
pip install git+https://github.com/pytorch/tnt.git@master
export UREG=<install-dir>/ureg/
````

# Training

To pre-train a model with STL10:
````bash
    ${UREG}/bin/pretrain.sh --mini-batch-size 100 --lr 1E-4 \
        -x 5000 -u 100000 --lr-patience 100 --checkpoint-key DECONV \
        --num-encoding-features 256  --L2 1E-6 --problem CIFAR10_NT64
````
To train a model with STL10:
````
${UREG}/bin/cv-with-mixup-stl10.sh -n 4000 -x 8000 -u 100000 \
   --mini-batch-size 100 --model PreActResNet18 --lr 0.01 \
   --alpha 0.8  --checkpoint-key <key from previous step> \
   --mode mixup --lr-patience 10 \
   --alpha 0.9 --unsup-proportion 0.5 \
   --L2 6.988858391214236E-4 \
   --cross-validation-indices 1 \
   --cross-validation-folds ${UREG}/data/stl10_binary/fold_indices.txt \
   --max-epochs 1000  --load-pre-trained-model
````

To train a modified GAN:

````bash
python bin/train_gan.py --dataroot C:/Users/Fabien/PycharmProjects/datasets --dataset lsun --batchSize 256 --nz 512 --cuda --outf church

````