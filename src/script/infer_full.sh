python infer_full.py \
    -opt options/stacked_model/stacked_model.yml \
    --phase1_path '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/src/exp_cls_256_cls_retrain_with9/UNI_lora_cls/_best.pt' \
    --phase2_path '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/src/exp_reorder_transformer_phase2/Transformer_Reorder/_best.pt' \
    --labels_dir '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/labels' \
    --images_dir '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/RAW_DATA/images'