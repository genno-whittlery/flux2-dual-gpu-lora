Write-Host "=== patchify fix marker ==="
Select-String -Path "C:\DiffSynth-Studio\diffsynth\models\wan_video_dit.py" -Pattern "return x, \(f, h, w\)" | Select-Object LineNumber, Line | Format-Table -AutoSize

Write-Host "=== dual-GPU helper file ==="
if (Test-Path "C:\DiffSynth-Studio\examples\wanvideo\model_training\wan_dual_gpu_diffsynth.py") {
    $item = Get-Item "C:\DiffSynth-Studio\examples\wanvideo\model_training\wan_dual_gpu_diffsynth.py"
    Write-Host "exists, $($item.Length) bytes, modified $($item.LastWriteTime)"
} else { Write-Host "MISSING" }

Write-Host "=== train.py dual-GPU wiring ==="
Select-String -Path "C:\DiffSynth-Studio\examples\wanvideo\model_training\train.py" -Pattern "WAN_DUAL_GPU|enable_wan_dual_gpu|FLUX2_DUAL_GPU" | Select-Object LineNumber, Line | Format-Table -AutoSize -Wrap

Write-Host "=== git status on DiffSynth-Studio ==="
Set-Location C:\DiffSynth-Studio
& git status --short 2>&1
Write-Host "branch: $(& git branch --show-current 2>&1)"
