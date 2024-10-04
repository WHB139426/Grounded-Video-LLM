# Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models
This is the repository for the video large langauge model : **Grounded-VideoLLM**.
<div align="center">
  <img src="model.png"/>
</div><br/>

## Install
1. Clone this repository and navigate to folder
```bash
git clone https://github.com/WHB139426/Grounded-Video-LLM.git
cd Grounded-Video-LLM
```

2. Install Package
```Shell
conda create -n grounded-videollm python=3.10.14
conda activate grounded-videollm
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 torchdata==0.8.0
pip install -r requirements.txt
```

**Some installation suggestions**
- We recommend you to pip install `flash-attn==2.3.3` and run the model with `torch.bfloat16`. If your device doesn't support these, you can skip them and replace the argparse parameter in `inference.py` by replacing `attn_implementation` and `dtype`, which may result in subtle numerical difference.

## Prepare the pretrained weights

Set your own `weight_path` to storage the pretrained weights. The folder should be organized as follows: 
```
├── Grounded-Video-LLM
│   └── inference.py
│   └── models
│   └── ...
├── weight_path
│   └── Phi-3.5-vision-instruct-seperated
│   └── internvideo
│   └── ckpt
│   └──...
```
Download the pretrained weights [[🤗HF](https://huggingface.co/WHB139426/Grounded-Video-LLM/tree/main)] in your own `weight_path`. 

## Qucik Inference
We give a short example to run the inference code. We recommend GPUs with 24GB memeroy.
1. replace the parameter `weight_path` in `scripts/inference.sh` with your own path that you set above.
2. run the following script: `bash scripts/inference.sh` to run our example.
3. You can change the `prompt_grounding`, `prompt_videoqa`, `video_path` in argparse and run `python inference.py` for your own case.

[![Watch the video](https://img.youtube.com/vi/_3klvlS4W7A/0.jpg)](https://www.youtube.com/watch?v=_3klvlS4W7A)


## TODO List
- [x] Release the Phi3.5-Vision-Instruct version.
- [ ] Release the LLaVA-Next-LLAMA3-8B version (coming soon).
- [ ] Release the training scripts and datasets.