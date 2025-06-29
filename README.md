## Structure
```
├── test_input                  <- Place the input images here
├── test_output                  <- The output results are saved here
├── infer.py                   <- Code for DNF-intrinsic prediction
├── requirement.txt       <- Env file
└── README.md
```

## Installation
```
pip install -r requirement.txt
```

## Downloading stable-diffusion-3 
Please manually download the Stable Diffusion 3 model weights from the official Hugging Face repository at: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/
```
├── stable-diffusion-3-medium-diffusers                  
├──── model_index.json
├──── vae
├──── transformer
├──── tokenizer
├──── tokenizer_2
├──── tokenizer_3
├──── text_encoder
├──── text_encoder_2
├──── text_encoder_3
├──── scheduler
```

## Downloading DNF-Intrinisc Lora model
Download the checkpoint folder from https://drive.google.com/drive/folders/1mTHbkTY-58iewCMp_KOxD5wq98PpwoJe?usp=drive_link
```
├── checkpoint                  
├──── adapter_config.json
├──── adapter_model.safetensors
```

## Inference
```
python infer.py \
--pretrained_model_path "./stable-diffusion-3-medium-diffusers/" \
--input_dir ./test_input \
--output_dir ./test_output \
--peft_model_path ./checkpoint \
--num_inference_steps 10 \
--device cuda:0
```
