python infer_full.py \
    -opt options/stacked_model/uni_lora_bbdm.yml \
    --phase1_path '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/src/exp/exp_cls_256_cls_retrain_with9/UNI_lora_cls/_best.pt' \
    --phase2_path '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/BBDM2/results/PredIm-2-ysubx/checkpoint/top_model_epoch_39.pth' \
    --labels_dir '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/labels' \
    --images_dir '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/images'