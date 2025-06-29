## Structure
```
├── test_input                  <- Place the input images here
├── test_output                  <- The output results are saved here
├── infer.py                   <- Code for DNF-intrinsic prediction
├── requirement.txt       <- Env file
└── README.md
```

## Installation
pip install -r requirement.txt

## Downloading stable-diffusion-3 checkpoint
Please manually download the Stable Diffusion 3 model weights from the official Hugging Face repository at: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/

## Inference
python infer.py \
--pretrained_model_path "./stable-diffusion-3-medium-diffusers/" \
--input_dir ./test_input \
--output_dir ./test_output \
--peft_model_path ./checkpoint \
--num_inference_steps 10 \
--device cuda:0
