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

$BundleFile = "$BinName-$Version-$Platform.zip"
$BundleUrl = "$BaseUrl/runtime/v$Version/$BundleFile"
$BinFile = "$BinName-$Version-$Platform.exe"
$DownloadUrl = "$BaseUrl/runtime/v$Version/$BinFile"

Write-Host "Installing kapsl $Version ($Platform) to $InstallDir..."

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) "kapsl-install-$([Guid]::NewGuid().ToString())"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

try {
    $TempBundle = Join-Path $TempDir $BundleFile
    $ExtractDir = Join-Path $TempDir "bundle"

    try {
        Invoke-WebRequest -Uri $BundleUrl -OutFile $TempBundle -UseBasicParsing
        New-Item -ItemType Directory -Path $ExtractDir -Force | Out-Null
        Expand-Archive -Path $TempBundle -DestinationPath $ExtractDir -Force

        $BundleExe = Get-ChildItem -Path $ExtractDir -Filter "$BinName.exe" -Recurse -File | Select-Object -First 1
        if (-not $BundleExe) {
            throw "Downloaded bundle does not contain $BinName.exe"
        }

        Copy-Item -Path (Join-Path $BundleExe.Directory.FullName "*") -Destination $InstallDir -Recurse -Force
    } catch {
        Write-Host "Bundle install failed or is unavailable: $($_.Exception.Message)"
        Write-Host "Falling back to single executable."
        $TempFile = Join-Path $TempDir $BinFile
        try {
            Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempFile -UseBasicParsing
        } catch {
            Write-Error "Download failed: $BundleUrl and $DownloadUrl"
            exit 1
        }

        $DestPath = Join-Path $InstallDir "$BinName.exe"
        Move-Item -Path $TempFile -Destination $DestPath -Force
    }
} finally {
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}

$DestPath = Join-Path $InstallDir "$BinName.exe"
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
