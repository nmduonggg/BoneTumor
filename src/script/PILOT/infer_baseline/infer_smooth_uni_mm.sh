python infer_smooth_mm.py \
    -opt options/UNI_lora/uni_lora_resnet_heavy_multimag.yml \
    --weight_path '/home/manhduong/BoneTumor/src/weights/UNI_lora_mm/_last_uni_resnet.pt' \
    --labels_dir '/workdir/radish/manhduong/labels' \
    --images_dir '/workdir/radish/manhduong/images' \
    --outdir './infer/smooth_uni_mm_pretrain/'