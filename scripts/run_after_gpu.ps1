# Runs after run_all_gpu
Set-Location "C:\Users\naman\Desktop\DL_FINAL FINAL"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"
while (-not (Select-String -Path data\interim\run_progress.log -Pattern "ALL DONE" -Quiet)) { Start-Sleep 30 }

function Step($name, $argv) {
    $t = Get-Date
    "$(Get-Date -Format HH:mm) START $name" | Add-Content data\interim\run_progress.log
    & $py @argv *> "data\interim\run_$name.log"
    "$(Get-Date -Format HH:mm) END   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)" | Add-Content data\interim\run_progress.log
}
Step "e7_llm"        @("scripts\e7_llm_zeroshot.py")
Step "e6_learning"   @("scripts\e6_learning_curve.py")
Step "h0_figures"    @("scripts\h0_figures.py")
Step "h1_errors"     @("scripts\h1_error_analysis.py")
Step "h2_deck"       @("scripts\h2_deck.py")
"$(Get-Date -Format HH:mm) AFTER DONE" | Add-Content data\interim\run_progress.log
