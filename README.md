# DNF-Intrinsic
1. pip install -r requirement.txt
2. python infer.py \
  --pretrained_model_path "./stable-diffusion-3-medium-diffusers/" \
  --input_dir ./test_input \
  --output_dir ./test_output \
  --peft_model_path ./checkpoint \
  --num_inference_steps 5 \
  --device cuda:0
