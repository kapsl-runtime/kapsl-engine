# Published Kapsl Windows installer with CUDA 12 support.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$source = Invoke-RestMethod -Uri "https://downloads.kapsl.net/install.ps1" -UseBasicParsing
$installer = [ScriptBlock]::Create([string]$source)
& $installer -Accelerator "cuda"
