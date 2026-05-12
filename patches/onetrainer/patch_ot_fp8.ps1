# Patch OneTrainer config: transformer weight_dtype INT_W8A8 -> FLOAT_8 (fp8 weight-only),
# matching the precision class of ai-toolkit / musubi / DiffSynth runs.
$cfgPath = "C:\OneTrainer\configs\sumi_dualgpu.json"
$cfg = Get-Content $cfgPath -Raw

# Replace only the transformer.weight_dtype value. Use a regex anchored on the
# transformer block to avoid touching prior.weight_dtype or text_encoder.weight_dtype.
$updated = $cfg -replace '("transformer":\s*\{\s*"train":\s*true,\s*"weight_dtype":\s*)"INT_W8A8"', '$1"FLOAT_8"'

if ($updated -eq $cfg) {
    Write-Host "PATTERN_NOT_FOUND or ALREADY_PATCHED"
    if ($cfg -match '"transformer":\s*\{\s*"train":\s*true,\s*"weight_dtype":\s*"FLOAT_8"') {
        Write-Host "  (config already set to FLOAT_8)"
    }
} else {
    $updated | Set-Content -Path $cfgPath -NoNewline
    Write-Host "PATCHED -- WROTE $cfgPath"
}

Write-Host ""
Write-Host "--- transformer block in updated config ---"
Get-Content $cfgPath | Select-String -Pattern '"transformer":' -Context 0,2
