# Sequential GPU pipeline for the real labels (Stages E-G)
Set-Location "C:\Users\naman\Desktop\DL_FINAL FINAL"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONWARNINGS = "ignore"
$py = ".venv\Scripts\python.exe"

function Step($name, $argv) {
    $t = Get-Date
    "$(Get-Date -Format HH:mm) START $name" | Add-Content data\interim\run_progress.log
    & $py @argv *> "data\interim\run_$name.log"
    "$(Get-Date -Format HH:mm) END   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)" | Add-Content data\interim\run_progress.log
}

Step "e1_text"        @("scripts\e1_text_models.py", "--min-class", "30", "--protocols", "cv5,val2024,test2025,lodo")
Step "e1_text_min20"  @("scripts\e1_text_models.py", "--min-class", "20", "--protocols", "cv5", "--out-dir", "results\stage_e\sensitivity_min20")
Step "e5_wavlm_ft"    @("scripts\e5_wavlm_finetune.py", "--protocols", "cv5,lodo")
Step "e6_learning"    @("scripts\e6_learning_curve.py")
Step "e4_t2_neural"   @("scripts\e4_t2_neural.py", "--min-class", "30")
Step "g0_t4"          @("scripts\g0_t4_mixed_effects.py")
"$(Get-Date -Format HH:mm) ALL DONE" | Add-Content data\interim\run_progress.log
