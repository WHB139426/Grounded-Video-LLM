weight_path='/home/haibo/weights'
data_dir='/home/haibo/data'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node 8 train.py \
    --model llava_next_video \
    --llm phi3.5 \
    --dataset mix_sft \
    --max_txt_len 2048 \
    --num_temporal_tokens 300 \
    --num_frames 96 \
    --num_segs 12 \
    --stage sft \
    --epoch 1 \
    --lora \
    --lora_lr 2e-4 \
    --lr 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear-warmup+cosine-decay \
    --sharding_strategy full-shard \
    --global_batch_size 96 \
    --per_device_batch_size 12 \
    --pretrained_proj ./experiments/grounded_llava_next_video_phi3.5_mix_grounded_multi_modal_projector_video_projecter_language_model.pth \
    --data_dir ${data_dir} \
    --save_dir ./experiments \
    --config_path ${weight_path}/Phi-3.5-vision-instruct \
    --tokenizer_path ${weight_path}/Phi-3.5-mini-instruct \
    --pretrained_video_path ${weight_path}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt \
    --pretrained_vision_proj_llm_path ${weight_path}/Phi-3.5-vision-instruct-seperated \