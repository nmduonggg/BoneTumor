python infer_full.py \
    -opt options/stacked_model/uni_lora_discrete_bbdm.yml \
    --phase1_path '/home/nmduongg/BoneTumor/works/BoneTumor/src/weights/UNI_lora_cls/_best.pt' \
    --phase2_path '/home/nmduongg/BoneTumor/works/BoneTumor/src/weights/diffusion_model/top_model_epoch_22.pth' \
    --labels_dir '/home/nmduongg/BoneTumor/works/DATA/labels' \
    --images_dir '/home/nmduongg/BoneTumor/works/DATA/images' \
    --outdir './infer/smooth_stacked_discrete_ensemble_not68/'