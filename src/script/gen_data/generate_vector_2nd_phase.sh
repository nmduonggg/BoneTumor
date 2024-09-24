python generate_vector_2nd_phase.py \
    -opt options/uni_lora_cls.yml \
    --weight_path '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/src/exp_cls_256_cls_retrain_with9/UNI_lora_cls/_best.pt' \
    --data_dir '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/phase2_training_data/' \
    --outdir '/mnt/disk1/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/RAW/REAL_WSIs/phase2_training_data/_VECTORS'