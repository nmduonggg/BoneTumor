python infer_smooth.py \
    -opt options/baselines/SwinV2_baseline.yml \
    --weight_path '/home/user01/aiotlab/nmduong/BoneTumor/src/exp_cls_256_cls_retrain_with9/SwinTransformerV2_baseline/_best.pt' \
    --labels_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/labels' \
    --images_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/images' \
    --outdir './infer/smooth_swinv2/'