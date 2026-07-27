use super::*;
use serde::Deserialize;

const DEFAULT_DOWNLOAD_BASE_URL: &str = "https://downloads.kapsl.net";
const DOWNLOAD_BASE_URL_ENV: &str = "KAPSL_BASE_URL";
const RELEASE_VERSION_ENV: &str = "KAPSL_VERSION";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderPack {
    Cuda12,
    TensorRt10,
}

impl ProviderPack {
    fn provider(self) -> &'static str {
        match self {
            Self::Cuda12 => "cuda",
            Self::TensorRt10 => "tensorrt",
        }
    }

    fn package_stem(self) -> &'static str {
        match self {
            Self::Cuda12 => "kapsl-provider-cuda12",
            Self::TensorRt10 => "kapsl-provider-tensorrt10",
        }
    }

    fn manifest_name(self) -> &'static str {
        match self {
            Self::Cuda12 => "kapsl-provider-cuda12.json",
            Self::TensorRt10 => "kapsl-provider-tensorrt10.json",
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Cuda12 => "CUDA 12",
            Self::TensorRt10 => "TensorRT 10",
        }
    }
}

#[derive(Debug, Deserialize)]
struct ProviderPackManifest {
    provider: String,
    runtime_version: String,
    platform: String,
    files: Vec<String>,
}

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn create() -> Result<Self, DynError> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kapsl-provider-install-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn execute_provider_command(args: ProviderCommandArgs) -> Result<(), DynError> {
    match args.command {
        ProviderSubcommand::Install(args) => execute_provider_install_command(args),
    }
}

fn execute_provider_install_command(args: ProviderInstallCommandArgs) -> Result<(), DynError> {
    if !cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        return Err(
            "`kapsl provider install` currently supports Windows x86_64. On Linux, install the matching provider .deb or tar.gz package from the release."
                .into(),
        );
    }

    let install_dir = match args.install_dir {
        Some(path) => path,
        None => std::env::current_exe()?
            .parent()
            .ok_or("Could not determine the directory containing kapsl.exe")?
            .to_path_buf(),
    };
    if !install_dir.is_dir() {
        return Err(format!(
            "The Kapsl installation directory does not exist: {}",
            install_dir.display()
        )
        .into());
    }

    let requested = match args.provider {
        ProviderPackage::Cuda12 => vec![ProviderPack::Cuda12],
        ProviderPackage::TensorRt10 => vec![ProviderPack::Cuda12, ProviderPack::TensorRt10],
    };
    let version = release_version();
    let base_url = std::env::var(DOWNLOAD_BASE_URL_ENV)
        .unwrap_or_else(|_| DEFAULT_DOWNLOAD_BASE_URL.to_string());

    for pack in requested {
        install_provider_pack(pack, &version, &base_url, &install_dir, args.force)?;
    }

    println!();
    println!("Provider installation complete.");
    match args.provider {
        ProviderPackage::Cuda12 => {
            println!("CUDA 12 is now available to Kapsl's automatic provider selection.");
        }
        ProviderPackage::TensorRt10 => {
            println!(
                "TensorRT 10 is now available to packages that declare `preferred_provider: tensorrt`."
            );
            println!(
                "Set KAPSL_PROVIDER_POLICY=manifest to require the package-declared provider."
            );
        }
    }
    Ok(())
}

fn install_provider_pack(
    pack: ProviderPack,
    version: &str,
    base_url: &str,
    install_dir: &Path,
    force: bool,
) -> Result<(), DynError> {
    if !force && installed_manifest_is_complete(pack, version, install_dir) {
        println!(
            "{} provider pack is already installed.",
            pack.display_name()
        );
        return Ok(());
    }

    let temporary = TemporaryDirectory::create()?;
    let archive_name = provider_archive_name(pack, version);
    let release_url = provider_release_url(base_url, version);
    let archive_url = format!("{release_url}/{archive_name}");
    let checksum_url = format!("{archive_url}.sha256");
    let archive_path = temporary.path().join(&archive_name);
    let extract_dir = temporary.path().join("extracted");
    fs::create_dir_all(&extract_dir)?;

    println!(
        "Downloading {} provider pack for Kapsl {}...",
        pack.display_name(),
        version
    );
    download_file(&archive_url, &archive_path)?;
    let expected_checksum = download_checksum(&checksum_url)?;
    let actual_checksum = sha256_file(&archive_path)?;
    if !actual_checksum.eq_ignore_ascii_case(&expected_checksum) {
        return Err(format!(
            "Checksum verification failed for {archive_name}: expected {expected_checksum}, got {actual_checksum}"
        )
        .into());
    }

    expand_zip_with_powershell(&archive_path, &extract_dir)?;
    let manifest_path = extract_dir.join(pack.manifest_name());
    let manifest = validate_staged_manifest(pack, version, &manifest_path, &extract_dir)?;

    deactivate_existing_manifests(pack, install_dir)?;
    for file_name in &manifest.files {
        fs::copy(extract_dir.join(file_name), install_dir.join(file_name)).map_err(|error| {
            format!(
                "Could not install {} into {}: {}. If Kapsl was installed system-wide, rerun this command from an Administrator PowerShell.",
                file_name,
                install_dir.display(),
                error
            )
        })?;
    }
    fs::copy(&manifest_path, install_dir.join(pack.manifest_name())).map_err(|error| {
        format!(
            "Could not activate the {} provider pack in {}: {}",
            pack.display_name(),
            install_dir.display(),
            error
        )
    })?;

    println!(
        "Installed {} into {}.",
        pack.display_name(),
        install_dir.display()
    );
    Ok(())
}

fn release_version() -> String {
    std::env::var(RELEASE_VERSION_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            option_env!("KAPSL_VERSION")
                .unwrap_or(env!("CARGO_PKG_VERSION"))
                .to_string()
        })
}

fn provider_release_url(base_url: &str, version: &str) -> String {
    let channel_path = if version.contains("-beta.") {
        "runtime/beta"
    } else {
        "runtime"
    };
    format!(
        "{}/{channel_path}/v{}",
        base_url.trim_end_matches('/'),
        version
    )
}

fn provider_archive_name(pack: ProviderPack, version: &str) -> String {
    format!("{}-{}-windows-x86_64.zip", pack.package_stem(), version)
}

fn download_file(url: &str, path: &Path) -> Result<(), DynError> {
    let agent = http_agent_for_transfer();
    let mut response = agent.get(url).call().map_err(|error| {
        format!(
            "Failed to download {url}: {}",
            format_remote_http_error(error)
        )
    })?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut reader = response.body_mut().as_reader();
    std::io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn download_checksum(url: &str) -> Result<String, DynError> {
    let agent = native_tls_http_agent();
    let mut response = agent.get(url).call().map_err(|error| {
        format!(
            "Failed to download checksum {url}: {}",
            format_remote_http_error(error)
        )
    })?;
    let content = response.body_mut().read_to_string()?;
    let checksum = content
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("Checksum file from {url} was empty"))?;
    if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("Checksum file from {url} did not contain a SHA-256 digest").into());
    }
    Ok(checksum.to_ascii_lowercase())
}

fn sha256_file(path: &Path) -> Result<String, DynError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn expand_zip_with_powershell(archive: &Path, destination: &Path) -> Result<(), DynError> {
    let command = "& { param([string]$Archive, [string]$Destination) Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force }";
    let status = Command::new("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ])
        .arg(archive)
        .arg(destination)
        .status()
        .map_err(|error| {
            format!("Could not start PowerShell to extract the provider pack: {error}")
        })?;
    if !status.success() {
        return Err(format!(
            "PowerShell failed to extract the provider pack (exit code {}).",
            status.code().unwrap_or(-1)
        )
        .into());
    }
    Ok(())
}

fn validate_staged_manifest(
    pack: ProviderPack,
    version: &str,
    manifest_path: &Path,
    directory: &Path,
) -> Result<ProviderPackManifest, DynError> {
    let manifest: ProviderPackManifest = serde_json::from_slice(&fs::read(manifest_path)?)
        .map_err(|error| {
            format!(
                "Invalid provider manifest {}: {error}",
                manifest_path.display()
            )
        })?;
    if !manifest.provider.eq_ignore_ascii_case(pack.provider()) {
        return Err(format!(
            "Provider manifest identifies `{}` instead of `{}`",
            manifest.provider,
            pack.provider()
        )
        .into());
    }
    if manifest.runtime_version != version {
        return Err(format!(
            "Provider pack targets Kapsl {}, but this runtime is {}",
            manifest.runtime_version, version
        )
        .into());
    }
    if manifest.platform != "windows-x86_64" {
        return Err(format!(
            "Provider pack targets {}, not windows-x86_64",
            manifest.platform
        )
        .into());
    }
    if manifest.files.is_empty() {
        return Err("Provider manifest contains no files".into());
    }
    for file_name in &manifest.files {
        let relative = Path::new(file_name);
        if relative.components().count() != 1 || relative.file_name().is_none() {
            return Err(format!("Unsafe provider manifest path: {file_name}").into());
        }
        if !directory.join(relative).is_file() {
            return Err(format!("Provider pack is missing declared file: {file_name}").into());
        }
    }
    Ok(manifest)
}

fn installed_manifest_is_complete(pack: ProviderPack, version: &str, directory: &Path) -> bool {
    let manifest_path = directory.join(pack.manifest_name());
    validate_staged_manifest(pack, version, &manifest_path, directory).is_ok()
}

fn deactivate_existing_manifests(pack: ProviderPack, directory: &Path) -> Result<(), DynError> {
    let prefix = pack.package_stem();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if entry.path().is_file() && name.starts_with(prefix) && name.ends_with(".json") {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_urls_follow_release_channel() {
        assert_eq!(
            provider_release_url("https://downloads.kapsl.net/", "0.1.18"),
            "https://downloads.kapsl.net/runtime/v0.1.18"
        );
        assert_eq!(
            provider_release_url(
                "https://downloads.kapsl.net",
                "0.1.18-beta.20260727.abcdef12"
            ),
            "https://downloads.kapsl.net/runtime/beta/v0.1.18-beta.20260727.abcdef12"
        );
    }

    #[test]
    fn provider_archive_names_are_versioned() {
        assert_eq!(
            provider_archive_name(ProviderPack::Cuda12, "0.1.18"),
            "kapsl-provider-cuda12-0.1.18-windows-x86_64.zip"
        );
        assert_eq!(
            provider_archive_name(ProviderPack::TensorRt10, "0.1.18"),
            "kapsl-provider-tensorrt10-0.1.18-windows-x86_64.zip"
        );
    }

    #[test]
    fn provider_install_command_uses_friendly_names() {
        let cli = Cli::try_parse_from(["kapsl", "provider", "install", "cuda12"])
            .expect("parse CUDA provider install");
        assert!(matches!(
            cli.command,
            Some(KapslCommand::Provider(ProviderCommandArgs {
                command: ProviderSubcommand::Install(ProviderInstallCommandArgs {
                    provider: ProviderPackage::Cuda12,
                    ..
                })
            }))
        ));

        let cli = Cli::try_parse_from(["kapsl", "provider", "install", "tensorrt"])
            .expect("parse TensorRT provider alias");
        assert!(matches!(
            cli.command,
            Some(KapslCommand::Provider(ProviderCommandArgs {
                command: ProviderSubcommand::Install(ProviderInstallCommandArgs {
                    provider: ProviderPackage::TensorRt10,
                    ..
                })
            }))
        ));
    }

    #[test]
    fn staged_manifest_must_match_runtime_and_platform() {
        let temporary = TemporaryDirectory::create().expect("create temporary directory");
        let directory = temporary.path();
        fs::write(
            directory.join("onnxruntime_providers_cuda.dll"),
            b"provider",
        )
        .expect("write provider fixture");
        let manifest_path = directory.join(ProviderPack::Cuda12.manifest_name());
        fs::write(
            &manifest_path,
            br#"{
                "provider":"cuda",
                "runtime_version":"0.1.18",
                "platform":"windows-x86_64",
                "files":["onnxruntime_providers_cuda.dll"]
            }"#,
        )
        .expect("write manifest fixture");

        assert!(validate_staged_manifest(
            ProviderPack::Cuda12,
            "0.1.18",
            &manifest_path,
            directory
        )
        .is_ok());
        assert!(validate_staged_manifest(
            ProviderPack::Cuda12,
            "0.1.19",
            &manifest_path,
            directory
        )
        .is_err());
    }
}
