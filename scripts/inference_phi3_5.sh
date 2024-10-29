weight_path='/home/haibo/weight_path'

python inference.py \
    --device cuda:0 \
    --llm phi3.5 \
    --config_path ${weight_path}/Phi-3.5-vision-instruct \
    --tokenizer_path ${weight_path}/Phi-3.5-mini-instruct \
    --pretrained_video_path ${weight_path}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt \
    --pretrained_vision_proj_llm_path ${weight_path}/Phi-3.5-vision-instruct-seperated \
    --ckpt_path ${weight_path}/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth \