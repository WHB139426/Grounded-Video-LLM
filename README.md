<h2 align="center"> <a href="https://arxiv.org/abs/2410.03290">Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models</a></h2>

🌟 This is the official repository for the video large langauge model : **Grounded-VideoLLM**, a Video-LLM adept at fine-grained temporal grounding. **Grounded-VideoLLM** not only excels in grounding tasks such as temporal sentence grounding, dense video captioning, and grounded VideoQA, but also shows great potential as a versatile video assistant for general video understanding.

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2410.03290-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.03290)
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/WHB139426/Grounded-Video-LLM/tree/main)

</h5>

<div align="center">
  <img src="model.png"/>
</div><br/>



💡 We sharpen our model by incorporating:
- An additional temporal stream to encode the relationships between frames. 
- Discrete temporal tokens enriched with specific time knowledge to represent timestamps. 
- A multi-stage training scheme, beginning with simple video-captioning tasks and progressively introducing video temporal grounding tasks of increasing complexity. To further enhance the temporal reasoning capability, we also curate a grounded VideoQA dataset by an automatic annotation pipeline. 

## 📰 News
- [x] **[2024.10.4]** Release the inference scripts and pretrained checkpoints.
- [x] **[2024.10.4]** Release the annotated grounded-VideoQA dataset .
- [x] **[2024.10.4]** Release the Phi3.5-Vision-Instruct version.
- [x] **[2024.10.29]** Release the LLaVA-Next-LLAMA3-8B version, with stronger performance in both grounding tasks and general benchmarks.
- [ ] Release the training scripts and training datasets. We will try to adapt more MLLMs as the base model for Grounded-VideoLLM in future.

## Performance
| Model Name                | LLM | Charades-STA (R1@0.3/R1@0.5/R1@0.7/mIoU) | ActivityNet-Groudning (R1@0.3/R1@0.5/R1@0.7/mIoU) | ActivityNet-Captions (SODA_c/METEOR) | NEXT-GQA (GQA/mIoP/mIoU) | MVbench | Video-MME (w/o subs) |
|---------------------------|-----|---------------------------------------|------------------------------------------------|--------------------------------------|-----------------------------|----------------------|----------------------|
| Grounded-VideoLLM         | Phi3.5-3.8B      | 54.2/36.4/19.7/36.8 | 46.2/30.3/19.0/36.1 | 6.0/6.8 | 26.7/34.5/21.1 | 59.4 | 47.7 |
| Grounded-VideoLLM (*)     | Phi3.5-3.8B      | 70.2/55.9/33.2/49.4 | 64.9/47.8/30.4/47.2 | 6.6/6.5 | 29.4/37.4/27.0 | 60.0 | 48.1 |
<!-- | Grounded-VideoLLM (*)     | LLaMA3-8B        | -                   | -                   | -       |  -             | -    | -    | -->
- (*) means we incorporate a sub training set of Charades-STA and ActivityNet into the third training stage. Please refer to our paper for more results.

## 🛠️ Install
1. Clone this repository and navigate to folder
```bash
git clone https://github.com/WHB139426/Grounded-Video-LLM.git
cd Grounded-Video-LLM
```

2. Install Package
```Shell
conda create -n grounded-videollm python=3.10.14
conda activate grounded-videollm
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 torchdata==0.8.0 # to make sure install torch before flash-attn
pip install -r requirements.txt
pip install numpy==1.26.4 # to make sure numpy<2.0
```

**Some installation suggestions**
- We recommend you to pip install `flash-attn==2.3.3` and run the model with `torch.bfloat16`. If your device doesn't support these, you can skip them and replace the argparse parameter `attn_implementation` and `dtype` in `inference.py`, which may result in subtle numerical difference.

## 🤗 Prepare the pretrained weights
Set your own `weight_path` to storage the pretrained weights. The folder should be organized as follows: 
```
├── Grounded-Video-LLM
│   └── inference.py
│   └── models
│   └── mm_utils
│   └── training
│   └── scripts
│   └── ...
├── weight_path
│   └── Phi-3.5-mini-instruct
│   └── Phi-3.5-vision-instruct-seperated
│   └── Phi-3.5-vision-instruct
│   └── llama3-llava-next-8b
│   └── llama3-llava-next-8b-seperated
│   └── Meta-Llama-3-8B-Instruct
│   └── ckpt
│   └── internvideo
│   └──...
```
Download the pretrained weights [[🤗HF](https://huggingface.co/WHB139426/Grounded-Video-LLM/tree/main)] in your own `weight_path`. 

## 🚀 Qucik Start
We give a brief example to run the inference code. We recommend GPUs with 24GB memeroy for Phi3.5 version, while 32GB memeroy for LLaVA-Next-LLAMA3-8B version.
1. replace the parameter `weight_path` in `scripts/inference_phi3_5.sh` or `scripts/inference_llama3.sh` with your own weight_path that you set above.
2. run the command `bash scripts/inference_phi3_5.sh` or `bash scripts/inference_llama3.sh` to reproduce the example below:

https://private-user-images.githubusercontent.com/115783170/373608088-69f83fd7-59d2-4105-a766-cd712d14d425.mp4

```
[USER] Give you a textual query: "The female host wearing purple clothes is reporting news in the studio". When does the described content occur in the video? Please return the start and end timestamps.
[Grounded-VideoLLM] From 14.20 seconds to 25.09 seconds.

[USER] Give you a textual query: "A sign written with 'NO TRESPASSING LOITERING DRUGS'". When does the described content occur in the video? Please return the start and end timestamps.
[Grounded-VideoLLM] From 107.95 seconds to 113.16 seconds.

[USER] What is happening from 70 seconds to 80 seconds?
[Grounded-VideoLLM] A woman with glasses and a red shirt is talking to a reporter.

[USER] Why was the man in green clothes interviewed?
[Grounded-VideoLLM] The man in green clothes was interviewed to provide his perspective on the incident and the history of violence in the apartment complex.

[USER] Question: What does this TV news report about?\nOptions:\n(A) thievery\n(B) community violence incidents\n(C) fashion show\n(D) aging population
[Grounded-VideoLLM] Answer: (B) community violence incidents
```
3. You can change the parameter of `prompt_grounding`, `prompt_videoqa`, `prompt_referring` and `video_path` in `inference.py`'s argparse to run your own case.

## 🎬 Grounded-VideoQA dataset
We provide the Grounded-VideoQA dataset that we annotated with GPT-4o-mini in [[🤗HF](https://huggingface.co/datasets/WHB139426/Grounded-VideoLLM/blob/main/G-VideoQA-gpt4o-mini-anno.json)]. You can download the videos following [[ActivityNet](https://activity-net.org/download.html)] and [[QVHighlights](https://github.com/jayleicn/moment_detr)].


## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{wang2024grounded,
  title={Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models},
  author={Wang, Haibo and Xu, Zhiyang and Cheng, Yu and Diao, Shizhe and Zhou, Yufan and Cao, Yixin and Wang, Qifan and Ge, Weifeng and Huang, Lifu},
  journal={arXiv preprint arXiv:2410.03290},
  year={2024}
}
```

## 🤝 Acknowledgement
We are grateful for the following awesome projects our Grounded-VideoLLM arising from: [Prismatic-VLMs](https://github.com/TRI-ML/prismatic-vlms), [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct), [InternVideo2](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2), [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT), [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat), [VTimeLLM](https://github.com/huangb23/VTimeLLM), [Momentor](https://github.com/DCDmllm/Momentor).
