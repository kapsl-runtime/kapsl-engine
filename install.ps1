# Kapsl CLI installer for Windows
# Usage: irm https://downloads.kapsl.net/install.ps1 | iex
# Optional NVIDIA pack: $env:KAPSL_ACCELERATOR="cuda" (or "tensorrt")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BaseUrl = if ($env:KAPSL_BASE_URL) { $env:KAPSL_BASE_URL } else { "https://downloads.kapsl.net" }
$BinName = "kapsl"
$InstallDir = if ($env:KAPSL_INSTALL_DIR) { $env:KAPSL_INSTALL_DIR } else { "$env:LOCALAPPDATA\Kapsl\bin" }
$Accelerator = if ($env:KAPSL_ACCELERATOR) { $env:KAPSL_ACCELERATOR.ToLowerInvariant() } else { "cpu" }

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

function Install-ProviderPack {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Provider,
        [Parameter(Mandatory = $true)]
        [string]$ProviderVersion,
        [Parameter(Mandatory = $true)]
        [string]$TempDirectory
    )

    if ($Platform -ne "windows-x86_64") {
        throw "The $Provider provider pack is currently available only for Windows x86_64."
    }

    $PackFile = "kapsl-provider-$Provider$ProviderVersion-$Version-$Platform.zip"
    $PackUrl = "$BaseUrl/runtime/v$Version/$PackFile"
    $PackArchive = Join-Path $TempDirectory $PackFile
    $PackExtract = Join-Path $TempDirectory "$Provider-provider"

    Write-Host "Installing Kapsl $Provider$ProviderVersion provider pack..."
    Invoke-WebRequest -Uri $PackUrl -OutFile $PackArchive -UseBasicParsing
    New-Item -ItemType Directory -Path $PackExtract -Force | Out-Null
    Expand-Archive -Path $PackArchive -DestinationPath $PackExtract -Force
    Copy-Item -Path (Join-Path $PackExtract "*") -Destination $InstallDir -Recurse -Force
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
$Platform = Get-Platform

if ($Accelerator -notin @("cpu", "cuda", "cuda12", "tensorrt", "tensorrt10")) {
    throw "Unsupported KAPSL_ACCELERATOR '$Accelerator'. Use cpu, cuda, or tensorrt."
}

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

    if ($Accelerator -in @("cuda", "cuda12", "tensorrt", "tensorrt10")) {
        Install-ProviderPack -Provider "cuda" -ProviderVersion "12" -TempDirectory $TempDir
    }
    if ($Accelerator -in @("tensorrt", "tensorrt10")) {
        Install-ProviderPack -Provider "tensorrt" -ProviderVersion "10" -TempDirectory $TempDir
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
if ($Accelerator -ne "cpu") {
    Write-Host "Installed accelerator profile: $Accelerator"
}
Write-Host "Run 'kapsl --help' to get started."
