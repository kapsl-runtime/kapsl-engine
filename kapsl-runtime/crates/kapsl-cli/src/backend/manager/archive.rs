//! Bounded, traversal-safe extraction of backend pack archives.

use super::*;

pub(crate) fn safe_extract_tar_gz(
    archive: &Path,
    destination: &Path,
    max_bytes: u64,
) -> ManagerResult<()> {
    let file = File::open(archive)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let mut total = 0_u64;
    for (position, entry) in archive
        .entries()
        .map_err(|error| BackendManagerError::new(format!("read backend archive: {error}")))?
        .enumerate()
    {
        if position >= MAX_ARCHIVE_ENTRIES {
            return Err(BackendManagerError::new(format!(
                "backend archive contains more than {MAX_ARCHIVE_ENTRIES} entries"
            )));
        }
        let mut entry = entry.map_err(|error| {
            BackendManagerError::new(format!("read backend archive entry: {error}"))
        })?;
        let relative = entry
            .path()
            .map_err(|error| BackendManagerError::new(error.to_string()))?
            .into_owned();
        if relative.is_absolute()
            || relative.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(BackendManagerError::new(format!(
                "backend archive contains unsafe path {}",
                relative.display()
            )));
        }
        let entry_type = entry.header().entry_type();
        if !(entry_type.is_file() || entry_type.is_dir() || entry_type.is_symlink()) {
            return Err(BackendManagerError::new(format!(
                "backend archive contains unsupported entry type for {}",
                relative.display()
            )));
        }
        total = total.saturating_add(entry.size());
        if total > max_bytes {
            return Err(BackendManagerError::new(format!(
                "backend archive expands beyond its signed {} byte limit",
                max_bytes
            )));
        }
        ensure_no_symlink_ancestor(destination, &relative)?;
        if !entry.unpack_in(destination).map_err(|error| {
            BackendManagerError::new(format!(
                "extract backend archive entry {}: {error}",
                relative.display()
            ))
        })? {
            return Err(BackendManagerError::new(format!(
                "backend archive entry escaped the staging directory: {}",
                relative.display()
            )));
        }
    }
    validate_tree_symlinks(destination, destination)?;
    Ok(())
}

pub(super) fn ensure_no_symlink_ancestor(root: &Path, relative: &Path) -> ManagerResult<()> {
    let mut current = root.to_path_buf();
    let mut components = relative.components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(component) = component else {
            continue;
        };
        if components.peek().is_none() {
            break;
        }
        current.push(component);
        if fs::symlink_metadata(&current)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Err(BackendManagerError::new(format!(
                "backend archive writes through symlink {}",
                current.display()
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_tree_symlinks(root: &Path, directory: &Path) -> ManagerResult<()> {
    for entry in read_dirs(directory)? {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            let target = fs::read_link(&path)?;
            if target.is_absolute() || !lexical_target_stays_inside(root, &path, &target) {
                return Err(BackendManagerError::new(format!(
                    "backend archive contains escaping symlink {} -> {}",
                    path.display(),
                    target.display()
                )));
            }
        } else if metadata.is_dir() {
            validate_tree_symlinks(root, &path)?;
        }
    }
    Ok(())
}

pub(super) fn lexical_target_stays_inside(root: &Path, link: &Path, target: &Path) -> bool {
    let Ok(relative_parent) = link.parent().unwrap_or(root).strip_prefix(root) else {
        return false;
    };
    let mut depth = 0_i64;
    for component in relative_parent.components().chain(target.components()) {
        match component {
            Component::CurDir => {}
            Component::Normal(_) => depth += 1,
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            Component::RootDir | Component::Prefix(_) => return false,
        }
    }
    true
}

pub(super) fn find_payload_root(root: &Path) -> ManagerResult<PathBuf> {
    fn visit(root: &Path, depth: usize, found: &mut Vec<PathBuf>) -> ManagerResult<()> {
        if depth > 4 {
            return Ok(());
        }
        if root.join(PAYLOAD_MANIFEST_NAME).is_file() {
            found.push(root.to_path_buf());
            return Ok(());
        }
        for entry in read_dirs(root)? {
            let path = entry.path();
            if path.is_dir() && !fs::symlink_metadata(&path)?.file_type().is_symlink() {
                visit(&path, depth + 1, found)?;
            }
        }
        Ok(())
    }

    let mut found = Vec::new();
    visit(root, 0, &mut found)?;
    match found.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(BackendManagerError::new(format!(
            "backend archive does not contain {PAYLOAD_MANIFEST_NAME}"
        ))),
        _ => Err(BackendManagerError::new(format!(
            "backend archive contains multiple {PAYLOAD_MANIFEST_NAME} files"
        ))),
    }
}
