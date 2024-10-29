weight_path='/home/haibo/weight_path'

python inference.py \
    --device cuda:0 \
    --llm llama3 \
    --config_path ${weight_path}/llama3-llava-next-8b \
    --tokenizer_path ${weight_path}/Meta-Llama-3-8B-Instruct \
    --pretrained_video_path ${weight_path}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt \
    --pretrained_vision_proj_llm_path ${weight_path}/llama3-llava-next-8b-seperated \
    --ckpt_path ${weight_path}/ckpt/sft_llava_next_video_llama3_mix_sft_multi_modal_projector_video_projecter_language_model.pth \