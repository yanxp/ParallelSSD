srun --partition=Bigvideo --mpi=pmi2 --gres=gpu:4 --job-name=ssd --kill-on-bad-exit=1 python train_RFB.py -d VOC -v RFB_E_vgg -s 512 -b 32 --ngpu 4
