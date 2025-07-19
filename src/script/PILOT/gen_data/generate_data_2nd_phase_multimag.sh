python generate_data_2nd_phase.py \
    -opt options/UNI_lora/uni_lora_cls_heavy_multimag_ISBI.yml \
    --weight_path '/home/user01/aiotlab/nmduong/BoneTumor/src/exp_cls_256_cls_retrain_with9_rm_confused_cases_multimag/UNI_lora_cls_multimag_ISBI/_best.pt' \
    --data_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/phase2_training_data' \
    --outdir '/home/user01/aiotlab/nmduong/BoneTumor/RAW/REAL_WSIs/_DENOISE2_multimag'