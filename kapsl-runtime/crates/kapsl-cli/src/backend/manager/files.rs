//! Filesystem and download helpers shared by the backend cache.

use super::*;

pub(super) fn lock_file(path: &Path) -> ManagerResult<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| {
            BackendManagerError::new(format!("open lock file {}: {error}", path.display()))
        })
}

pub(super) fn process_path_lock(path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks
        .entry(path.to_path_buf())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

pub(super) fn read_file_bounded(path: &Path, limit: u64) -> ManagerResult<Vec<u8>> {
    let file = File::open(path)
        .map_err(|error| BackendManagerError::new(format!("open {}: {error}", path.display())))?;
    if file.metadata()?.len() > limit {
        return Err(BackendManagerError::new(format!(
            "{} exceeds the {} byte limit",
            path.display(),
            limit
        )));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(BackendManagerError::new(format!(
            "{} exceeds the {} byte limit",
            path.display(),
            limit
        )));
    }
    Ok(bytes)
}

pub(super) fn read_json_bounded<T: for<'de> Deserialize<'de>>(
    path: &Path,
    limit: u64,
) -> ManagerResult<T> {
    let bytes = read_file_bounded(path, limit)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        BackendManagerError::new(format!("invalid JSON in {}: {error}", path.display()))
    })
}

pub(super) fn download_bytes(url: &str, limit: u64) -> ManagerResult<Vec<u8>> {
    let agent = http_client::http_agent_for_transfer();
    let mut response = agent.get(url).call().map_err(|error| {
        BackendManagerError::new(format!(
            "download {url}: {}",
            http_client::format_remote_http_error(error)
        ))
    })?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(BackendManagerError::new(format!(
            "download from {url} exceeds the {limit} byte limit"
        )));
    }
    Ok(bytes)
}

pub(crate) fn sha256_file(path: &Path) -> ManagerResult<String> {
    Ok(checksums::sha256_file(path)?)
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> ManagerResult<()> {
    let parent = path.parent().ok_or_else(|| {
        BackendManagerError::new(format!("{} has no parent directory", path.display()))
    })?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name().and_then(OsStr::to_str).unwrap_or("file"),
        unique_nonce()
    ));
    {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    if path.exists() {
        quarantine_file(path)?;
    }
    fs::rename(&temporary, path)?;
    sync_parent(path)
}

pub(super) fn quarantine_file(path: &Path) -> ManagerResult<()> {
    if !path.exists() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| BackendManagerError::new(format!("{} has no parent", path.display())))?;
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or("file");
    let destination = parent.join(format!(".{name}.invalid-{}", unique_nonce()));
    fs::rename(path, destination)?;
    Ok(())
}

pub(super) fn unique_nonce() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

pub(super) fn sync_parent(path: &Path) -> ManagerResult<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub(super) fn read_dirs(path: &Path) -> ManagerResult<Vec<fs::DirEntry>> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| {
            BackendManagerError::new(format!("read directory {}: {error}", path.display()))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BackendManagerError::new(error.to_string()))?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

pub(super) fn directory_size(path: &Path) -> ManagerResult<u64> {
    let mut total = 0_u64;
    for entry in read_dirs(path)? {
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            total = total.saturating_add(directory_size(&entry.path())?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}
