srun --partition=Bigvideo --mpi=pmi2 --gres=gpu:1 --job-name=Yolo --kill-on-bad-exit=1 python demo_logo.py -d Logo -v RFB_E_vgg -s 512 --trained_model logo149RFB_E_vgg_Logo_epoches_65.pth --demo data/Logo/testJPEG/burgerkingimg000377.jpg