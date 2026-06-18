use super::*;

pub(crate) struct EngineRoutes {
    pub(crate) reader: warp::filters::BoxedFilter<(warp::reply::Response,)>,
    pub(crate) admin: warp::filters::BoxedFilter<(warp::reply::Response,)>,
}

pub(crate) fn build_engine_routes() -> EngineRoutes {
    // Engine packaging + remote registry endpoints.
    let package_kapsl = warp::path!("api" / "engine" / "package")
        .and(warp::post())
        .and(warp::body::json())
        .map(|request: PackageKapslRequest| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            match create_kapsl_package(&request, false) {
                Ok(response) => {
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                }
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse { error }),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let push_kapsl = warp::path!("api" / "engine" / "push")
        .and(warp::post())
        .and(warp::body::json())
        .map(|request: PushKapslRequest| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            match push_kapsl_to_placeholder_remote(&request) {
                Ok(response) => {
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                }
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse { error }),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let pull_kapsl = warp::path!("api" / "engine" / "pull")
        .and(warp::post())
        .and(warp::body::json())
        .map(|request: PullKapslRequest| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            match pull_kapsl_from_placeholder_remote(&request) {
                Ok(response) => {
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                }
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse { error }),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let list_remote_artifacts = warp::path!("api" / "engine" / "remote-artifacts")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .map(|query: HashMap<String, String>| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            match fetch_remote_artifact_inventory(query.get("remote_url").map(String::as_str)) {
                Ok(response) => {
                    warp::reply::with_status(warp::reply::json(&response), StatusCode::OK)
                }
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse { error }),
                    StatusCode::BAD_GATEWAY,
                ),
            }
        });

    // File browser — lets the web UI navigate the server filesystem to pick model paths.
    let browse_fs = warp::path!("api" / "engine" / "browse")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .map(|query: HashMap<String, String>| {
            use std::path::Path;
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct BrowseEntry {
                name: String,
                path: String,
                is_dir: bool,
                #[serde(skip_serializing_if = "Option::is_none")]
                size: Option<u64>,
            }

            #[derive(Serialize)]
            struct BrowseResponse {
                path: String,
                entries: Vec<BrowseEntry>,
            }

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            let requested = query.get("path").map(|s| s.as_str()).unwrap_or("").trim();

            let dir = if requested.is_empty() {
                dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/"))
            } else {
                std::path::PathBuf::from(requested)
            };

            if !dir.exists() || !dir.is_dir() {
                return warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: format!("Not a directory: {}", dir.display()),
                    }),
                    StatusCode::BAD_REQUEST,
                );
            }

            let canonical = match dir.canonicalize() {
                Ok(p) => p,
                Err(e) => {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Cannot resolve path: {e}"),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
            };

            let mut entries: Vec<BrowseEntry> = Vec::new();

            // Parent directory entry (omit at filesystem root)
            if let Some(parent) = canonical.parent() {
                entries.push(BrowseEntry {
                    name: "..".to_string(),
                    path: parent.to_string_lossy().into_owned(),
                    is_dir: true,
                    size: None,
                });
            }

            let mut read_entries: Vec<_> = match std::fs::read_dir(&canonical) {
                Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
                Err(e) => {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Cannot read directory: {e}"),
                        }),
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }
            };
            read_entries.sort_by_key(|e| {
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                (!is_dir, e.file_name())
            });

            for entry in read_entries {
                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(_) => continue,
                };
                let name = entry.file_name().to_string_lossy().into_owned();
                // Skip hidden files (dotfiles)
                if name.starts_with('.') {
                    continue;
                }
                let path = entry.path().to_string_lossy().into_owned();
                let is_dir = file_type.is_dir();
                let size = if is_dir {
                    None
                } else {
                    entry.metadata().ok().map(|m| m.len())
                };
                // For files, only show model-relevant extensions
                if !is_dir {
                    let ext = Path::new(&name)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if !matches!(
                        ext.as_str(),
                        "aimod" | "kapsl" | "gguf" | "onnx" | "bin" | "safetensors"
                    ) {
                        continue;
                    }
                }
                entries.push(BrowseEntry {
                    name,
                    path,
                    is_dir,
                    size,
                });
            }

            warp::reply::with_status(
                warp::reply::json(&BrowseResponse {
                    path: canonical.to_string_lossy().into_owned(),
                    entries,
                }),
                StatusCode::OK,
            )
        });

    let reader = list_remote_artifacts.map(reply_into_response).boxed();
    let admin = browse_fs
        .or(package_kapsl)
        .or(push_kapsl)
        .or(pull_kapsl)
        .map(reply_into_response)
        .boxed();

    EngineRoutes { reader, admin }
}
