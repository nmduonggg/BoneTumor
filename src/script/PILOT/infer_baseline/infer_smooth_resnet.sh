python infer_smooth.py \
    -opt options/ResNet_baseline.yml \
    --weight_path '/home/user01/aiotlab/nmduong/BoneTumor/src/exp_cls_256_cls_retrain_with9/ResNet_baseline/_best.pt' \
    --labels_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/labels' \
    --images_dir '/home/user01/aiotlab/nmduong/BoneTumor/RAW_DATA/images' \
    --outdir './infer/smooth_resnet/'