$f = "C:\DiffSynth-Studio\diffsynth\models\wan_video_dit.py"
Write-Host "=== patchify body ==="
$content = Get-Content $f
$inPatchify = $false
$inForward = $false
$forwardLines = 0
for ($i = 0; $i -lt $content.Length; $i++) {
    $line = $content[$i]
    if ($line -match "    def patchify\(") {
        $inPatchify = $true
        Write-Host "$($i+1): $line"
        continue
    }
    if ($inPatchify) {
        if ($line -match "    def " -and -not ($line -match "patchify")) {
            $inPatchify = $false
        } else {
            Write-Host "$($i+1): $line"
        }
    }
}
Write-Host ""
Write-Host "=== forward() near patchify call ==="
for ($i = 0; $i -lt $content.Length; $i++) {
    if ($content[$i] -match "self\.patchify\(x\)") {
        for ($j = [math]::Max(0, $i-1); $j -lt [math]::Min($content.Length, $i+5); $j++) {
            Write-Host "$($j+1): $($content[$j])"
        }
        break
    }
}

Write-Host ""
Write-Host "=== git status ==="
Set-Location C:\DiffSynth-Studio
& git status --short diffsynth/models/wan_video_dit.py 2>&1
Write-Host "branch: $(& git branch --show-current 2>&1)"
Write-Host "head: $(& git log -1 --format='%h %s' 2>&1)"
