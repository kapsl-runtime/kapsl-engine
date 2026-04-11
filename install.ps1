# Kapsl CLI installer for Windows
# Usage: irm https://downloads.kapsl.net/install.ps1 | iex
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BaseUrl = if ($env:KAPSL_BASE_URL) { $env:KAPSL_BASE_URL } else { "https://downloads.kapsl.net" }
$BinName = "kapsl"
$InstallDir = if ($env:KAPSL_INSTALL_DIR) { $env:KAPSL_INSTALL_DIR } else { "$env:LOCALAPPDATA\Kapsl\bin" }

# ---------------------------------------------------------------------------
# Detect architecture
# ---------------------------------------------------------------------------
function Get-Platform {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64"   { return "windows-x86_64" }
        "Arm64" { return "windows-aarch64" }
        default {
            Write-Error "Unsupported architecture: $arch"
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# Resolve latest version from R2
# ---------------------------------------------------------------------------
function Get-LatestVersion {
    $url = "$BaseUrl/runtime/latest.txt"
    try {
        return (Invoke-RestMethod -Uri $url -UseBasicParsing).Trim()
    } catch {
        Write-Error "Failed to fetch latest version from $url"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
$Platform = Get-Platform

$Version = $env:KAPSL_VERSION
if (-not $Version) {
    Write-Host "Fetching latest version... " -NoNewline
    $Version = Get-LatestVersion
    Write-Host $Version
}

$BinFile = "$BinName-$Version-$Platform.exe"
$DownloadUrl = "$BaseUrl/runtime/v$Version/$BinFile"

Write-Host "Installing kapsl $Version ($Platform) to $InstallDir..."

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

$TempFile = Join-Path ([System.IO.Path]::GetTempPath()) "$BinFile"
try {
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempFile -UseBasicParsing
} catch {
    Write-Error "Download failed: $DownloadUrl"
    exit 1
}

$DestPath = Join-Path $InstallDir "$BinName.exe"
Move-Item -Path $TempFile -Destination $DestPath -Force

Write-Host "Installed to $DestPath"

# Add to user PATH if not already present
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$UserPath;$InstallDir", "User")
    Write-Host ""
    Write-Host "Added $InstallDir to your user PATH."
    Write-Host "Restart your terminal for the change to take effect."
}

Write-Host ""
Write-Host "Run 'kapsl --help' to get started."
