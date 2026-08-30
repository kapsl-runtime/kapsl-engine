# Published latest Kapsl beta Windows installer.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$source = Invoke-RestMethod -Uri "https://downloads.kapsl.net/install-beta-base.ps1" -UseBasicParsing
$installer = [ScriptBlock]::Create([string]$source)
& $installer -Channel "beta"
