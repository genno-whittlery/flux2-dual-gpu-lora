# Wan 2.2 I2V A14B (high_noise) dual-GPU bf16 smoke train.
# Uses --model_paths (JSON) to point DiffSynth at our local files directly,
# bypassing modelscope cache lookup that doesn't find F:\models\Wan-AI\... .
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:WAN_DUAL_GPU = "true"
$env:CUDA_VISIBLE_DEVICES = "0,1"

Set-Location C:\DiffSynth-Studio

$base = "F:/models/Wan-AI/Wan2.2-I2V-A14B"
$paths = @(
    "$base/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
    "$base/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
    "$base/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
    "$base/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
    "$base/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
    "$base/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
    "$base/models_t5_umt5-xxl-enc-bf16.pth",
    "$base/Wan2.1_VAE.pth"
)
$modelPathsJson = $paths | ConvertTo-Json -Compress

Write-Host "=== model_paths JSON ==="
Write-Host $modelPathsJson
Write-Host ""
Write-Host "=== weight presence ==="
foreach ($p in $paths) {
    $winPath = $p -replace "/", "\"
    if (Test-Path $winPath) { Write-Host "OK $winPath" } else { Write-Host "MISSING $winPath" }
}

Write-Host ""
Write-Host "=== launching dual-GPU bf16 train ==="
& C:\DiffSynth-Studio\.venv\Scripts\accelerate.exe launch --num_processes 1 examples/wanvideo/model_training/train.py `
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B `
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-I2V-A14B/metadata.csv `
  --height 480 `
  --width 832 `
  --num_frames 49 `
  --dataset_repeat 5 `
  --num_epochs 1 `
  --model_paths $modelPathsJson `
  --learning_rate 1e-4 `
  --remove_prefix_in_ckpt "pipe.dit." `
  --output_path "./models/train/wan22_dualgpu_smoke" `
  --lora_base_model "dit" `
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" `
  --lora_rank 32 `
  --extra_inputs "input_image" `
  --max_timestep_boundary 0.358 `
  --min_timestep_boundary 0 `
  --use_gradient_checkpointing `
  --initialize_model_on_cpu 2>&1 | Tee-Object -FilePath C:\tmp\wan_dualgpu_smoke_v3.log

Write-Host "EXIT_CODE=$LASTEXITCODE"
