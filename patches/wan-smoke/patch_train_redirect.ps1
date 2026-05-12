$path = "C:\DiffSynth-Studio\examples\wanvideo\model_training\train.py"
$content = Get-Content $path -Raw

$old = 'self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)'
$new = 'self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config, redirect_common_files=False)'

if ($content -match [regex]::Escape($new)) {
    Write-Host "already patched"
} elseif ($content -match [regex]::Escape($old)) {
    $content = $content.Replace($old, $new)
    Set-Content -Path $path -Value $content -NoNewline
    Write-Host "patched"
} else {
    Write-Host "did not find target line; need manual patch"
    exit 1
}

Write-Host ""
Write-Host "=== verify ==="
Select-String -Path $path -Pattern "from_pretrained|redirect_common_files" | Select-Object LineNumber, Line | Format-Table -AutoSize -Wrap
