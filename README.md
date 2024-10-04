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
2. run the following script: `bash scripts/inference.sh`
3. You can change the `prompt_grounding`, `prompt_videoqa`, `video_path` in argparse to run your own case.

## TODO List
- [x] Release the Phi3.5-Vision-Instruct Version.
- [ ] Release the LLaVA-Next-LLAMA3-8B Version (soon).
- [ ] Release the training scripts and datasets.