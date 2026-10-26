# Render every slide of a
param([string]$Pptx = "reports\F1_radio_presentation.pptx", [string]$Out = "reports\pptx_render")
Set-Location "C:\Users\naman\Desktop\DL_FINAL FINAL"
$full = (Resolve-Path $Pptx).Path
New-Item -ItemType Directory -Force $Out | Out-Null
Get-ChildItem $Out -Filter *.png | Remove-Item -Force
$outFull = (Resolve-Path $Out).Path
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open($full, $true, $false, $false)
foreach ($s in $pres.Slides) { $s.Export(("{0}\s{1:D2}.png" -f $outFull, $s.SlideIndex), "PNG", 1600, 900) }
"rendered " + $pres.Slides.Count + " slides to $outFull"
$pres.Close(); $pp.Quit()
