# Installation

````
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision
pip install git+https://github.com/pytorch/tnt.git@master
export UREG=<install-dir>/ureg/
````

# Training

To train a model with STL10:
````
${UREG}/bin/cv-with-mixup-stl10.sh -n 4000 -x 8000 -u 100000 \
   --mini-batch-size 100 --model PreActResNet18 --lr 0.01 \
   --alpha 0.8  --checkpoint-key UNSUP_MIXUP \
   --mode mixup --lr-patience 10 \
   --alpha 0.9 --unsup-proportion 0.5 \
   --L2 6.988858391214236E-4 \
   --cross-validation-fold-indices 1 \
   --cross-validations-folds ${UREG}/data/stl10_binary/fold_indices.txt \
   --max-epochs 1000
````