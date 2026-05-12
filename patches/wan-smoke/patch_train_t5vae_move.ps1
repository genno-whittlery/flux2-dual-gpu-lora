$path = "C:\DiffSynth-Studio\examples\wanvideo\model_training\train.py"
$content = Get-Content $path -Raw

# Find the FLUX2_DUAL_GPU set line and inject T5/VAE move BEFORE it.
$marker = '        # Signal launch_training_task to skip its own model.to() move'
$injection = @'
        # After the dual-GPU DiT split, move non-DiT pipeline components
        # to cuda:0 explicitly. Without this, T5 text_encoder stays on
        # CPU and the first training step crashes with cross-device
        # index_select. FLUX.2 path doesn't need this because Mistral
        # is pre-cached via cache_text_embeddings; Wan encodes T5
        # on-the-fly during the training step.
        for _attr in ('text_encoder', 'vae', 'image_encoder'):
            _sub = getattr(model.pipe, _attr, None)
            if _sub is not None:
                _sub.to('cuda:0')
                print(f"[wan-dual-gpu] moved pipe.{_attr} to cuda:0")
'@

if ($content -match "moved pipe\.\{_attr\} to cuda:0") {
    Write-Host "already patched"
} elseif ($content.Contains($marker)) {
    $content = $content.Replace($marker, "$injection`r`n$marker")
    Set-Content -Path $path -Value $content -NoNewline
    Write-Host "patched"
} else {
    Write-Host "marker not found"
    exit 1
}

Write-Host ""
Write-Host "=== verify ==="
Select-String -Path $path -Pattern "moved pipe|text_encoder|FLUX2_DUAL_GPU" | Select-Object LineNumber, Line | Format-Table -AutoSize -Wrap
