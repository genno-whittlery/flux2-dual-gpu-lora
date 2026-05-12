"""Wan 2.2 I2V A14B dual-GPU bf16 smoke train launcher (v5).

Uses --model_id_with_origin_paths so DiffSynth handles shard discovery.
Files are accessed via a junction at C:\\DiffSynth-Studio\\models\\Wan-AI\\Wan2.2-I2V-A14B
pointing at F:\\models\\Wan-AI\\Wan2.2-I2V-A14B. Maverick's train.py has been
patched to pass redirect_common_files=False so T5/VAE .pth files aren't
redirected to the .safetensors mirror.
"""
import os
import subprocess
import sys

# Model ID + sharded patterns. DiffSynth's snapshot_download finds them
# in C:\DiffSynth-Studio\models\Wan-AI\Wan2.2-I2V-A14B (junction -> F:\models\...).
model_id_with_origin_paths = ",".join([
    "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors",
    "Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth",
    "Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth",
])

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
env["TOKENIZERS_PARALLELISM"] = "false"
env["WAN_DUAL_GPU"] = "true"
env["CUDA_VISIBLE_DEVICES"] = "0,1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cmd = [
    r"C:\DiffSynth-Studio\.venv\Scripts\accelerate.exe", "launch",
    "--num_processes", "1",
    "examples/wanvideo/model_training/train.py",
    "--dataset_base_path", "data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B",
    "--dataset_metadata_path", "data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B/metadata.csv",
    "--height", "480",
    "--width", "832",
    "--num_frames", "49",
    "--dataset_repeat", "5",
    "--num_epochs", "1",
    "--model_id_with_origin_paths", model_id_with_origin_paths,
    "--learning_rate", "1e-4",
    "--remove_prefix_in_ckpt", "pipe.dit.",
    "--output_path", "./models/train/wan22_dualgpu_smoke",
    "--lora_base_model", "dit",
    "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2",
    "--lora_rank", "32",
    "--extra_inputs", "input_image",
    "--max_timestep_boundary", "0.358",
    "--min_timestep_boundary", "0",
    "--use_gradient_checkpointing",
    "--use_gradient_checkpointing_offload",
    "--initialize_model_on_cpu",
]

print(f"=== model_id_with_origin_paths ===\n{model_id_with_origin_paths}\n")
print("=== launching ===", flush=True)
proc = subprocess.run(cmd, cwd=r"C:\DiffSynth-Studio", env=env)
print(f"\nEXIT_CODE={proc.returncode}")
sys.exit(proc.returncode)
