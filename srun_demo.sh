srun --partition=Bigvideo --mpi=pmi2 --gres=gpu:1 --job-name=Yolo --kill-on-bad-exit=1 python demo.py --cfg=./experiments/cfgs/rfb_resnet50_train_logo.yml --demo=1172400914.jpg
