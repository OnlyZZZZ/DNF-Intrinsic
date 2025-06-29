import os
import random
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from torchvision import transforms
from torchvision.transforms import Resize
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
import argparse


to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
resize_480 = Resize([480, 640])


def import_model_class(pretrained_model_name_or_path, subfolder="text_encoder", revision=None):
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class_name = config.architectures[0]
    if model_class_name == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class_name == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported text encoder: {model_class_name}")

def load_text_encoders(cls1, cls2, cls3, model_path):
    enc1 = cls1.from_pretrained(model_path, subfolder="text_encoder")
    enc2 = cls2.from_pretrained(model_path, subfolder="text_encoder_2")
    enc3 = cls3.from_pretrained(model_path, subfolder="text_encoder_3")
    return enc1, enc2, enc3

def _encode_with_t5(text_encoder, tokenizer, prompt, max_len, device):
    inputs = tokenizer(prompt, padding="max_length", max_length=max_len, truncation=True,
                       add_special_tokens=True, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    embeddings = text_encoder(ids)[0].to(dtype=text_encoder.dtype, device=device)
    return embeddings

def _encode_with_clip(text_encoder, tokenizer, prompt, device):
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    output = text_encoder(ids, output_hidden_states=True)
    prompt_embeds = output.hidden_states[-2].to(dtype=torch.float16, device=device)
    pooled = output[0]
    return prompt_embeds, pooled

def encode_prompt(prompt, encoders, tokenizers, max_len, device):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_embeds, pooled_embeds = [], []
    for i in range(2):
        clip_embed, pooled = _encode_with_clip(encoders[i], tokenizers[i], prompt, device)
        clip_embeds.append(clip_embed)
        pooled_embeds.append(pooled)

    clip_embed_cat = torch.cat(clip_embeds, dim=-1)
    pooled_embed_cat = torch.cat(pooled_embeds, dim=-1)

    t5_embed = _encode_with_t5(encoders[2], tokenizers[2], prompt, max_len, device)

    clip_embed_cat = torch.nn.functional.pad(clip_embed_cat,
                                             (0, t5_embed.shape[-1] - clip_embed_cat.shape[-1]))
    combined = torch.cat([clip_embed_cat, t5_embed], dim=-2)
    return combined, pooled_embed_cat

def compute_text_embeddings(prompt, encoders, tokenizers, device):
    with torch.no_grad():
        prompt_embed, pooled_embed = encode_prompt(prompt, encoders, tokenizers, max_len=77, device=device)
    return prompt_embed.to(device), pooled_embed.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNF-intrinsic Prediction Script")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to SD3 pretrained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to PEFT/LoRA weights")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument("--image_suffix", type=str, default=".png")
    args = parser.parse_args()
    
    device = "cuda:0"
    weight_dtype = torch.float16
    pretrained_model_path = args.pretrained_model_path
    output_dir = args.output_dir 
    data_root = args.input_dir 
    save_path = args.peft_model_path 
    num_inference_steps = args.num_inference_steps 
    image_suffix = args.image_suffix 
    
    # Tokenizers
    tokenizer1 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
    tokenizer3 = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_3")

    # Text Encoders
    enc_cls1 = import_model_class(pretrained_model_path)
    enc_cls2 = import_model_class(pretrained_model_path, subfolder="text_encoder_2")
    enc_cls3 = import_model_class(pretrained_model_path, subfolder="text_encoder_3")
    enc1, enc2, enc3 = load_text_encoders(enc_cls1, enc_cls2, enc_cls3, pretrained_model_path)

    tokenizers = [tokenizer1, tokenizer2, tokenizer3]
    text_encoders = [enc1, enc2, enc3]

    # Scheduler and main models
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer")

    # Load LoRA
    transformer = PeftModel.from_pretrained(transformer, save_path, device_map=device)

    for model in [vae, transformer, enc1, enc2, enc3]:
        model.requires_grad_(False)
        model.to(device, dtype=weight_dtype)

    os.makedirs(output_dir, exist_ok=True)
    data_list = [f[:-4] for f in os.listdir(data_root) if f.endswith(image_suffix)]
    targets = ['albedo', 'normal', 'depth', 'metallic', 'roughness']

    with torch.no_grad():
        for img_name in tqdm(data_list):
            image = Image.open(os.path.join(data_root, img_name + image_suffix)).convert("RGB")
            image_tensor = resize_480(to_tensor(image).unsqueeze(0)) * 2 - 1
            latents = vae.encode(image_tensor.to(device, dtype=vae.dtype)).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            for target in targets:
                prompt_embed, pooled_embed = compute_text_embeddings(target, text_encoders, tokenizers, device)

                scheduler.set_timesteps(num_inference_steps, device=device)
                current_latents = latents
                for t in scheduler.timesteps:
                    timestep = t.expand(current_latents.shape[0])
                    model_pred = transformer(
                        hidden_states=current_latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embed,
                        pooled_projections=pooled_embed,
                        return_dict=False
                    )[0]
                    current_latents = scheduler.step(model_pred, t, current_latents, return_dict=False)[0]

                img_latents = (current_latents / vae.config.scaling_factor) + vae.config.shift_factor
                output = vae.decode(img_latents.to(device, dtype=vae.dtype)).sample
                output = (output / 2 + 0.5).clamp(0, 1)

                if target in ["depth", "metallic", "roughness"]:
                    output = output.mean(dim=1, keepdim=True)

                out_img = to_pil(output[0].cpu())
                out_img.save(f"{output_dir}/{img_name}_{target}.png")
