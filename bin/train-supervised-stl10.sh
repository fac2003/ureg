# Run evaluate3

python src/org/campagnelab/dl/pytorch/cifar10/Evaluate3.py \
-n 5000 -x 8000 -u 100000 --mini-batch-size 128 --checkpoint-key STL10_SUPERVISED_FULL \
--lr 0.1  --num-epochs 10000 --problem STL10 \
--model VGG16