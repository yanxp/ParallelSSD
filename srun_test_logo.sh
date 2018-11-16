srun --partition=SenseMediaA --mpi=pmi2 --gres=gpu:1 --job-name=Yolo --kill-on-bad-exit=1 python test_logo.py -d Logo -v RFB_E_vgg -s 512 --trained_model aliyunRFB_E_vgg_Logo_epoches_45.pth

