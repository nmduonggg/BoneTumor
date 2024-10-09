python infer_smooth.py \
    -opt options/ResNet_baseline.yml \
    --weight_path '/mnt/disk4/nmduong/Vin-Uni-Bone-Tumor/BoneTumor/src/exp/exp_cls_256_cls_retrain_with9/ResNet_baseline/_best.pt' \
    --labels_dir '/mnt/disk4/nmduong/Vin-Uni-Bone-/labels' \
    --images_dir '/mnt/disk4/nmduong/Vin-Uni-Bone-/images' \
    --outdir './infer/smooth_resnet/'