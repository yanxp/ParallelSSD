srun --partition=SenseMediaF --mpi=pmi2 --gres=gpu:1 --job-name=Yolo --kill-on-bad-exit=1 python demo_logo.py -d Logo -v RFB_E_vgg -s 512 --trained_model logo38RFB_E_vgg_Logo_epoches_155.pth
