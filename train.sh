python train.py --weight_decay 0.0001 --pro 0.3 --alphas '0.1, 0.5, 1' --ps '32, 16' --search_T 4 --updata_epoch 50 --mix_alpha 1.0 --patch_size 16 --epochs 240 --lr_decay_epochs '150,180,210' --path_t './pretrained/wrn_40_2.pth' --model_s wrn_16_2 --kd_T 4 --trial 0

