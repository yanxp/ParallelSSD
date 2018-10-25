srun --partition=Bigvideo --mpi=pmi2 --gres=gpu:1 --job-name=Yolo --kill-on-bad-exit=1 python test_RFB.py -d VOC -v RFB_E_vgg -s 512 --trained_model weights/RFB_E_vgg_VOC_epoches_80.pth 
