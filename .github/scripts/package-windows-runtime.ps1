param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$NumericVersion,
    [Parameter(Mandatory = $true)]
    [string]$ProductName,
    [Parameter(Mandatory = $true)]
    [string]$UpgradeCode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-Architecture {
    $runnerArch = $env:RUNNER_ARCH.ToLowerInvariant()
    switch ($runnerArch) {
        "x64" {
            return @{
                Artifact = "x86_64"
                Wix = "x64"
                ProgramFiles = "ProgramFiles64Folder"
                Win64 = "yes"
            }
        }
        "arm64" {
            return @{
                Artifact = "aarch64"
                Wix = "arm64"
                ProgramFiles = "ProgramFiles64Folder"
                Win64 = "yes"
            }
        }
        default {
            return @{
                Artifact = $runnerArch
                Wix = "x86"
                ProgramFiles = "ProgramFilesFolder"
                Win64 = "no"
            }
        }
    }
}

function Find-OrtBundleDirectory {
    $searchRoots = @(
        "kapsl-runtime/target/release",
        "kapsl-runtime/target/release/deps"
    )
    if ($env:LOCALAPPDATA) {
        $searchRoots += Join-Path $env:LOCALAPPDATA "ort.pyke.io\dfbin"
    }
    if ($env:CARGO_HOME) {
        $searchRoots += $env:CARGO_HOME
    } elseif ($env:USERPROFILE) {
        $searchRoots += Join-Path $env:USERPROFILE ".cargo"
    }
    $searchRoots = @($searchRoots | Where-Object { Test-Path $_ })

    $sharedProvider = Get-ChildItem `
        -Path $searchRoots `
        -Filter "onnxruntime_providers_shared.dll" `
        -Recurse `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $sharedProvider) {
        throw "Could not find onnxruntime_providers_shared.dll after the runtime build."
    }

    return $sharedProvider.Directory.FullName
}

function Write-FileManifest {
    param(
        [string]$Directory,
        [string]$Provider,
        [string]$ProviderVersion,
        [string[]]$Requires
    )

    $files = @(
        Get-ChildItem -Path $Directory -File |
        Sort-Object Name |
        ForEach-Object { $_.Name }
    )
    $manifest = [ordered]@{
        schema_version = 1
        provider = $Provider
        provider_version = $ProviderVersion
        runtime_version = $Version
        platform = "windows-$($script:Architecture.Artifact)"
        ort_version = "1.23.2"
        requires = $Requires
        files = $files
    }
    $manifestPath = Join-Path $Directory "kapsl-provider-$Provider$ProviderVersion.json"
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding utf8
}

function New-CoreMsi {
    param([string]$SourceDirectory)

    $files = @(Get-ChildItem -Path $SourceDirectory -File | Sort-Object Name)
    if ($files.Count -eq 0) {
        throw "No files were staged for the Windows core installer."
    }

    $componentLines = @()
    $componentRefLines = @()
    for ($index = 0; $index -lt $files.Count; $index++) {
        $componentId = "RuntimeFileComponent$index"
        $fileId = "RuntimeFile$index"
        $fileName = $files[$index].Name
        $componentLines += "          <Component Id=""$componentId"" Guid=""*"" Win64=""$($script:Architecture.Win64)"">"
        $componentLines += "            <File Id=""$fileId"" Source=""$fileName"" KeyPath=""yes"" />"
        if ($fileName -eq "kapsl.exe") {
            $componentLines += '            <Environment Id="KapslPathEnv" Name="PATH" Value="[INSTALLFOLDER]" Action="set" Part="last" Permanent="no" System="yes" />'
        }
        $componentLines += '          </Component>'
        $componentRefLines += "      <ComponentRef Id=""$componentId"" />"
    }

    $wxsLines = @(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">'
        '  <Product'
        '    Id="*"'
        "    Name=""$ProductName"""
        '    Language="1033"'
        "    Version=""$NumericVersion"""
        '    Manufacturer="Kapsl"'
        "    UpgradeCode=""$UpgradeCode"">"
        '    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" />'
        "    <MajorUpgrade DowngradeErrorMessage=""A newer version of $ProductName is already installed."" />"
        '    <MediaTemplate EmbedCab="yes" />'
        '    <Directory Id="TARGETDIR" Name="SourceDir">'
        "      <Directory Id=""$($script:Architecture.ProgramFiles)"">"
        '        <Directory Id="INSTALLFOLDER" Name="Kapsl">'
    ) + $componentLines + @(
        '        </Directory>'
        '      </Directory>'
        '    </Directory>'
        "    <Feature Id=""MainFeature"" Title=""$ProductName"" Level=""1"">"
    ) + $componentRefLines + @(
        '    </Feature>'
        '  </Product>'
        '</Wix>'
    )

    $wxsPath = Join-Path $SourceDirectory "installer.wxs"
    $wxsLines | Set-Content -Path $wxsPath -Encoding utf8

    $msiName = "kapsl-runtime-$Version-windows-$($script:Architecture.Artifact).msi"
    $msiPath = Join-Path (Resolve-Path "dist").Path $msiName
    Push-Location $SourceDirectory
    try {
        & candle.exe -arch $script:Architecture.Wix "installer.wxs"
        if ($LASTEXITCODE -ne 0) {
            throw "candle.exe failed with exit code $LASTEXITCODE"
        }
        & light.exe "installer.wixobj" -o $msiPath
        if ($LASTEXITCODE -ne 0) {
            throw "light.exe failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $hash = (Get-FileHash -Path $msiPath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $msiName" | Set-Content -Path "$msiPath.sha256"
}

function Get-DependencyNames {
    param([string]$Path)

    $output = & $script:Dumpbin /DEPENDENTS $Path 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin.exe failed while inspecting $Path"
    }

    return @(
        $output |
        ForEach-Object {
            if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') {
                $Matches[1].ToLowerInvariant()
            }
        } |
        Sort-Object -Unique
    )
}

function Find-Dumpbin {
    $command = Get-Command "dumpbin.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "dumpbin.exe is not on PATH and vswhere.exe was not found."
    }
    $visualStudio = (& $vswhere -latest -products * -property installationPath).Trim()
    $msvcRoot = Join-Path $visualStudio "VC\Tools\MSVC"
    $msvcVersion = Get-ChildItem -Path $msvcRoot -Directory |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $msvcVersion) {
        throw "No MSVC tools installation was found under $msvcRoot."
    }
    $path = Join-Path $msvcVersion.FullName "bin\Hostx64\x64\dumpbin.exe"
    if (-not (Test-Path $path)) {
        throw "dumpbin.exe was not found at $path."
    }
    return $path
}

function Get-DependencyClosure {
    param(
        [System.IO.FileInfo[]]$Roots,
        [hashtable]$Candidates
    )

    $selected = @{}
    $queue = [System.Collections.Generic.Queue[System.IO.FileInfo]]::new()
    foreach ($root in $Roots) {
        $queue.Enqueue($root)
    }

    while ($queue.Count -gt 0) {
        $file = $queue.Dequeue()
        $key = $file.Name.ToLowerInvariant()
        if ($selected.ContainsKey($key)) {
            continue
        }
        $selected[$key] = $file

        foreach ($dependency in (Get-DependencyNames -Path $file.FullName)) {
            if ($Candidates.ContainsKey($dependency) -and -not $selected.ContainsKey($dependency)) {
                $queue.Enqueue($Candidates[$dependency])
            }
        }
    }

    return $selected
}

function Get-RequiredCandidate {
    param(
        [hashtable]$Candidates,
        [string]$Name
    )

    $key = $Name.ToLowerInvariant()
    if (-not $Candidates.ContainsKey($key)) {
        throw "NVIDIA dependency collection is missing required file: $Name"
    }
    return $Candidates[$key]
}

function New-ProviderArchive {
    param(
        [string]$Provider,
        [string]$ProviderVersion,
        [hashtable]$Files,
        [hashtable]$ExcludedFiles,
        [string[]]$Requires
    )

    $packDirectory = "provider-packs/$Provider$ProviderVersion"
    New-Item -ItemType Directory -Path $packDirectory -Force | Out-Null

    foreach ($entry in ($Files.GetEnumerator() | Sort-Object Key)) {
        if ($ExcludedFiles -and $ExcludedFiles.ContainsKey($entry.Key)) {
            continue
        }
        Copy-Item $entry.Value.FullName (Join-Path $packDirectory $entry.Value.Name) -Force
    }

    $licenseIndex = 0
    foreach ($license in $script:NvidiaLicenseFiles) {
        $licenseIndex++
        $extension = $license.Extension
        if (-not $extension) {
            $extension = ".txt"
        }
        Copy-Item `
            $license.FullName `
            (Join-Path $packDirectory "NVIDIA-LICENSE-$licenseIndex$extension") `
            -Force
    }

    Write-FileManifest `
        -Directory $packDirectory `
        -Provider $Provider `
        -ProviderVersion $ProviderVersion `
        -Requires $Requires

    $archiveName = "kapsl-provider-$Provider$ProviderVersion-$Version-windows-$($script:Architecture.Artifact).zip"
    $archivePath = Join-Path (Resolve-Path "dist").Path $archiveName
    if (Test-Path $archivePath) {
        Remove-Item $archivePath -Force
    }
    Push-Location $packDirectory
    try {
        & 7z.exe a -tzip $archivePath "*" -mx=5
        if ($LASTEXITCODE -ne 0) {
            throw "7z.exe failed to create $archiveName with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    $hash = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $archiveName" | Set-Content -Path "$archivePath.sha256"
}

$script:Architecture = Get-Architecture
New-Item -ItemType Directory -Path "dist" -Force | Out-Null
New-Item -ItemType Directory -Path "wix/core" -Force | Out-Null
New-Item -ItemType Directory -Path "provider-packs" -Force | Out-Null

Copy-Item "kapsl-runtime/target/release/kapsl.exe" "wix/core/kapsl.exe" -Force

$ortBundleDirectory = Find-OrtBundleDirectory
$ortDlls = @(Get-ChildItem -Path $ortBundleDirectory -Filter "*.dll" -File | Sort-Object Name)
foreach ($dll in $ortDlls) {
    if ($dll.Name -match '^onnxruntime_providers_(cuda|tensorrt)\.dll$') {
        continue
    }
    Write-Host "Staging core ONNX Runtime DLL: $($dll.Name)"
    Copy-Item $dll.FullName (Join-Path "wix/core" $dll.Name) -Force
}

if (-not (Test-Path "wix/core/onnxruntime_providers_shared.dll")) {
    throw "The Windows core installer is missing onnxruntime_providers_shared.dll."
}

New-CoreMsi -SourceDirectory "wix/core"

if ($env:RUNNER_ARCH -ne "X64") {
    Write-Host "NVIDIA provider packs are currently published only for Windows x86_64."
    exit 0
}

$nvidiaDependencies = Join-Path $env:RUNNER_TEMP "kapsl-nvidia-runtime-dlls"
if (Test-Path $nvidiaDependencies) {
    Remove-Item $nvidiaDependencies -Recurse -Force
}
New-Item -ItemType Directory -Path $nvidiaDependencies -Force | Out-Null

python -m pip install --upgrade pip
python -m pip install `
    --target $nvidiaDependencies `
    --only-binary=:all: `
    --extra-index-url https://pypi.nvidia.com `
    "onnxruntime-gpu[cuda,cudnn]==1.23.2" `
    "tensorrt-cu12-libs==10.16.1.11"

$candidates = @{}
foreach ($dll in (Get-ChildItem -Path $nvidiaDependencies -Filter "*.dll" -Recurse -File)) {
    $key = $dll.Name.ToLowerInvariant()
    if (-not $candidates.ContainsKey($key)) {
        $candidates[$key] = $dll
    }
}
foreach ($dll in $ortDlls) {
    $candidates[$dll.Name.ToLowerInvariant()] = $dll
}
$script:NvidiaLicenseFiles = @(
    Get-ChildItem -Path $nvidiaDependencies -Recurse -File |
    Where-Object { $_.Name -match '^(LICENSE|License|THIRD_PARTY_NOTICES)(\..+)?$' } |
    Sort-Object FullName
)
$script:Dumpbin = Find-Dumpbin

$cudaRootNames = @(
    "onnxruntime_providers_cuda.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cudnn64_9.dll",
    "cufft64_11.dll",
    "curand64_10.dll"
)
$cudaRoots = @($cudaRootNames | ForEach-Object {
    Get-RequiredCandidate -Candidates $candidates -Name $_
})
$cudaRoots += @(
    $candidates.GetEnumerator() |
    Where-Object { $_.Key -match '^cudnn.*64_9\.dll$' } |
    ForEach-Object { $_.Value }
)
$cudaFiles = Get-DependencyClosure -Roots $cudaRoots -Candidates $candidates

$coreFiles = @{}
foreach ($file in (Get-ChildItem -Path "wix/core" -Filter "*.dll" -File)) {
    $coreFiles[$file.Name.ToLowerInvariant()] = $file
}
New-ProviderArchive `
    -Provider "cuda" `
    -ProviderVersion "12" `
    -Files $cudaFiles `
    -ExcludedFiles $coreFiles `
    -Requires @()

$tensorRtRootNames = @(
    "onnxruntime_providers_tensorrt.dll",
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll"
)
$tensorRtRoots = @($tensorRtRootNames | ForEach-Object {
    Get-RequiredCandidate -Candidates $candidates -Name $_
})
$tensorRtRoots += @(
    $candidates.GetEnumerator() |
    Where-Object { $_.Key -match '^(nvinfer|nvonnxparser).*\.dll$' } |
    ForEach-Object { $_.Value }
)
$tensorRtFiles = Get-DependencyClosure -Roots $tensorRtRoots -Candidates $candidates

$tensorRtExcluded = @{}
foreach ($entry in $coreFiles.GetEnumerator()) {
    $tensorRtExcluded[$entry.Key] = $entry.Value
}
foreach ($entry in $cudaFiles.GetEnumerator()) {
    $tensorRtExcluded[$entry.Key] = $entry.Value
}
New-ProviderArchive `
    -Provider "tensorrt" `
    -ProviderVersion "10" `
    -Files $tensorRtFiles `
    -ExcludedFiles $tensorRtExcluded `
    -Requires @("cuda12")

Get-ChildItem -Path "dist" -File | Sort-Object Name
