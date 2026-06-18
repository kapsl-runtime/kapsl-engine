use super::*;

pub(crate) fn build_extension_routes(
    extension_manager_for_api: Arc<ExtensionManager>,
    running_connectors_for_api: Arc<
        AsyncMutex<HashMap<String, ConnectorClient<ConnectorRuntimeHandle>>>,
    >,
    rag_state_for_api: RagRuntimeState,
) -> warp::filters::BoxedFilter<(warp::reply::Response,)> {
    // Extensions API
    #[derive(Deserialize)]
    struct InstallExtensionRequest {
        path: Option<String>,
        extension_id: Option<String>,
        marketplace_url: Option<String>,
    }

    #[derive(Deserialize)]
    struct ExtensionConfigRequest {
        workspace_id: String,
        config: serde_json::Value,
    }

    #[derive(Deserialize)]
    struct LaunchExtensionRequest {
        workspace_id: String,
    }

    let extension_manager_for_list = extension_manager_for_api.clone();
    let list_extensions = warp::path!("api" / "extensions")
        .and(warp::get())
        .map(move || {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            let extensions = match extension_manager_for_list.registry.discover() {
                Ok(list) => list,
                Err(err) => {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: err.to_string(),
                        }),
                        StatusCode::INTERNAL_SERVER_ERROR,
                    );
                }
            };

            let payload: Vec<_> = extensions
                .into_iter()
                .map(|ext| {
                    json!({
                        "manifest": ext.manifest,
                        "path": ext.path.to_string_lossy()
                    })
                })
                .collect();

            warp::reply::with_status(warp::reply::json(&payload), StatusCode::OK)
        });

    let list_marketplace_extensions = warp::path!("api" / "extensions" / "marketplace")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .map(move |query: HashMap<String, String>| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            let q = query.get("q").map(String::as_str);
            let marketplace_url = query.get("marketplace_url").map(String::as_str);

            match fetch_extension_marketplace(q, marketplace_url) {
                Ok(payload) => {
                    warp::reply::with_status(warp::reply::json(&payload), StatusCode::OK)
                }
                Err(error) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse { error }),
                    StatusCode::BAD_GATEWAY,
                ),
            }
        });

    let extension_manager_for_install = extension_manager_for_api.clone();
    let install_extension = warp::path!("api" / "extensions" / "install")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |req: InstallExtensionRequest| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            let path = req.path.as_deref().unwrap_or("").trim().to_string();
            let extension_id = req.extension_id.as_deref().unwrap_or("").trim().to_string();

            if path.is_empty() && extension_id.is_empty() {
                return warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: "Provide either `path` or `extension_id`".to_string(),
                    }),
                    StatusCode::BAD_REQUEST,
                );
            }

            let install_result: Result<InstalledExtension, String> = if !path.is_empty() {
                let path = PathBuf::from(&path);
                if !path.exists() {
                    return warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!("Extension path does not exist: {}", path.display()),
                        }),
                        StatusCode::BAD_REQUEST,
                    );
                }
                extension_manager_for_install
                    .registry
                    .install_from_dir(&path)
                    .map_err(|e| e.to_string())
            } else {
                install_extension_from_marketplace(
                    &extension_manager_for_install.registry,
                    &extension_id,
                    req.marketplace_url.as_deref(),
                )
            };

            match install_result {
                Ok(ext) => warp::reply::with_status(
                    warp::reply::json(&json!({
                        "status": "ok",
                        "extension": {
                            "manifest": ext.manifest,
                            "path": ext.path.to_string_lossy()
                        }
                    })),
                    StatusCode::OK,
                ),
                Err(err) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: err.to_string(),
                    }),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let extension_manager_for_uninstall = extension_manager_for_api.clone();
    let uninstall_extension = warp::path!("api" / "extensions" / String / "uninstall")
        .and(warp::post())
        .map(move |extension_id: String| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            match extension_manager_for_uninstall
                .registry
                .uninstall(&extension_id)
            {
                Ok(()) => warp::reply::with_status(
                    warp::reply::json(&json!({ "status": "ok" })),
                    StatusCode::OK,
                ),
                Err(err) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: err.to_string(),
                    }),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let extension_manager_for_config = extension_manager_for_api.clone();
    let set_extension_config = warp::path!("api" / "extensions" / String / "config")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |extension_id: String, req: ExtensionConfigRequest| {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            match extension_manager_for_config.set_workspace_config(
                &req.workspace_id,
                &extension_id,
                &req.config,
            ) {
                Ok(()) => warp::reply::with_status(
                    warp::reply::json(&json!({ "status": "ok" })),
                    StatusCode::OK,
                ),
                Err(err) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: err.to_string(),
                    }),
                    StatusCode::BAD_REQUEST,
                ),
            }
        });

    let extension_manager_for_get_config = extension_manager_for_api.clone();
    let get_extension_config = warp::path!("api" / "extensions" / String / "config")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .map(
            move |extension_id: String, query: HashMap<String, String>| {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let workspace_id = match query.get("workspace_id") {
                    Some(id) if !id.trim().is_empty() => id.to_string(),
                    _ => {
                        return warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: "workspace_id query parameter is required".to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        );
                    }
                };

                match extension_manager_for_get_config
                    .get_workspace_config(&workspace_id, &extension_id)
                {
                    Ok(config) => warp::reply::with_status(
                        warp::reply::json(&json!({ "config": config })),
                        StatusCode::OK,
                    ),
                    Err(err) => warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: err.to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    ),
                }
            },
        );

    let extension_manager_for_launch = extension_manager_for_api.clone();
    let running_connectors_for_launch = running_connectors_for_api.clone();
    let launch_extension = warp::path!("api" / "extensions" / String / "launch")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |extension_id: String, req: LaunchExtensionRequest| {
            let extension_manager = extension_manager_for_launch.clone();
            let running_connectors = running_connectors_for_launch.clone();
            async move {
                use warp::http::StatusCode;

                #[derive(Serialize)]
                struct ErrorResponse {
                    error: String,
                }

                let key = extension_key(&req.workspace_id, &extension_id);
                {
                    let running = running_connectors.lock().await;
                    if running.contains_key(&key) {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: "connector already running".to_string(),
                            }),
                            StatusCode::CONFLICT,
                        ));
                    }
                }

                let extensions = match extension_manager.registry.discover() {
                    Ok(list) => list,
                    Err(err) => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: err.to_string(),
                            }),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        ));
                    }
                };
                let extension = match extensions
                    .into_iter()
                    .find(|ext| ext.manifest.id == extension_id)
                {
                    Some(ext) => ext,
                    None => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Extension {} not found", extension_id),
                            }),
                            StatusCode::NOT_FOUND,
                        ));
                    }
                };

                let mut client =
                    match extension_manager.launch_connector(&req.workspace_id, &extension) {
                        Ok(client) => client,
                        Err(err) => {
                            return Ok(warp::reply::with_status(
                                warp::reply::json(&ErrorResponse {
                                    error: err.to_string(),
                                }),
                                StatusCode::BAD_REQUEST,
                            ));
                        }
                    };

                let config = match extension_manager
                    .get_workspace_connector_config(&req.workspace_id, &extension_id)
                {
                    Ok(config) => config.unwrap_or_else(|| json!({})),
                    Err(err) => {
                        let _ = client.shutdown();
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: err.to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }
                };

                let response = match client.request(ConnectorRequestKind::ValidateConfig { config })
                {
                    Ok(response) => response,
                    Err(err) => {
                        let _ = client.shutdown();
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: err.to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }
                };

                if let ConnectorResponseKind::Err(err) = response.kind {
                    let _ = client.shutdown();
                    return Ok(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse { error: err.message }),
                        StatusCode::BAD_REQUEST,
                    ));
                }

                let mut running = running_connectors.lock().await;
                if running.contains_key(&key) {
                    let _ = client.shutdown();
                    return Ok(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: "connector already running".to_string(),
                        }),
                        StatusCode::CONFLICT,
                    ));
                }
                running.insert(key, client);

                Ok::<_, warp::Rejection>(warp::reply::with_status(
                    warp::reply::json(&json!({ "status": "ok" })),
                    StatusCode::OK,
                ))
            }
        });

    let extension_manager_for_sync = extension_manager_for_api.clone();
    let running_connectors_for_sync = running_connectors_for_api.clone();
    let rag_state_for_sync = rag_state_for_api.clone();
    let sync_extension = warp::path!("api" / "extensions" / String / "sync")
    .and(warp::post())
    .and(warp::body::json())
    .and_then(move |extension_id: String, req: SyncExtensionRequest| {
        let extension_manager = extension_manager_for_sync.clone();
        let running_connectors = running_connectors_for_sync.clone();
        let rag_state = rag_state_for_sync.clone();
        async move {
            use warp::http::StatusCode;

            #[derive(Serialize)]
            struct ErrorResponse {
                error: String,
            }

            let workspace_id = req.workspace_id.trim().to_string();
            if workspace_id.is_empty() {
                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: "workspace_id is required".to_string(),
                    }),
                    StatusCode::BAD_REQUEST,
                ));
            }

            let connector_config = match extension_manager
                .get_workspace_connector_config(&workspace_id, &extension_id)
            {
                Ok(Some(config)) => config,
                Ok(None) => {
                    return Ok(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: format!(
                                "No connector config found for workspace `{}`",
                                workspace_id
                            ),
                        }),
                        StatusCode::BAD_REQUEST,
                    ));
                }
                Err(err) => {
                    return Ok(warp::reply::with_status(
                        warp::reply::json(&ErrorResponse {
                            error: err.to_string(),
                        }),
                        StatusCode::BAD_REQUEST,
                    ));
                }
            };

            let key = extension_key(&workspace_id, &extension_id);
            let (source_id, deltas, upsert_payloads, delete_doc_ids, fetch_failures) = {
                let mut running = running_connectors.lock().await;
                let client = match running.get_mut(&key) {
                    Some(client) => client,
                    None => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!(
                                    "Connector `{}` is not running for workspace `{}`",
                                    extension_id, workspace_id
                                ),
                            }),
                            StatusCode::CONFLICT,
                        ));
                    }
                };

                let source_id = match select_sync_source_id(
                    req.source_id.clone(),
                    connector_config.clone(),
                    client,
                ) {
                    Ok(source_id) => source_id,
                    Err(error) => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse { error }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }
                };

                let sync_response = match client.request(ConnectorRequestKind::Sync {
                    source_id: source_id.clone(),
                    cursor: req.cursor.clone(),
                }) {
                    Ok(response) => response,
                    Err(error) => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: format!("Sync request failed: {}", error),
                            }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }
                };

                let deltas: Vec<DocumentDelta> = match sync_response.kind {
                    ConnectorResponseKind::Err(err) => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse { error: err.message }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }
                    ConnectorResponseKind::Ok(ConnectorResult::Deltas(deltas)) => deltas,
                    _ => {
                        return Ok(warp::reply::with_status(
                            warp::reply::json(&ErrorResponse {
                                error: "connector returned unexpected Sync response"
                                    .to_string(),
                            }),
                            StatusCode::BAD_REQUEST,
                        ));
                    }
                };

                let mut upsert_payloads = Vec::new();
                let mut delete_doc_ids = Vec::new();
                let mut fetch_failures = 0usize;
                for delta in &deltas {
                    match delta.op {
                        DeltaOp::Delete => delete_doc_ids.push(delta.id.clone()),
                        DeltaOp::Upsert => {
                            let response = match client.request(
                                ConnectorRequestKind::FetchDocument {
                                    document_id: delta.id.clone(),
                                },
                            ) {
                                Ok(response) => response,
                                Err(error) => {
                                    fetch_failures += 1;
                                    log::warn!(
                                        "FetchDocument request failed: extension_id={} workspace_id={} source_id={} doc_id={} error={}",
                                        extension_id,
                                        workspace_id,
                                        source_id,
                                        delta.id,
                                        error
                                    );
                                    continue;
                                }
                            };

                            match response.kind {
                                ConnectorResponseKind::Err(err) => {
                                    fetch_failures += 1;
                                    log::warn!(
                                        "FetchDocument rejected by connector: extension_id={} workspace_id={} source_id={} doc_id={} error={}",
                                        extension_id,
                                        workspace_id,
                                        source_id,
                                        delta.id,
                                        err.message
                                    );
                                }
                                ConnectorResponseKind::Ok(ConnectorResult::Document(
                                    document,
                                )) => {
                                    upsert_payloads.push(document);
                                }
                                _ => {
                                    fetch_failures += 1;
                                    log::warn!(
                                        "FetchDocument returned unexpected response: extension_id={} workspace_id={} source_id={} doc_id={}",
                                        extension_id,
                                        workspace_id,
                                        source_id,
                                        delta.id
                                    );
                                }
                            }
                        }
                    }
                }

                (
                    source_id,
                    deltas,
                    upsert_payloads,
                    delete_doc_ids,
                    fetch_failures,
                )
            };

            let tenant_id = normalize_tenant_id(req.tenant_id.as_deref());
            let mut deleted = 0usize;
            let mut upserted = 0usize;
            let mut skipped_non_text = 0usize;
            let mut failed = fetch_failures;
            let mut chunk_count = 0usize;

            for doc_id in delete_doc_ids {
                match delete_document_from_rag(
                    &rag_state,
                    &tenant_id,
                    &workspace_id,
                    &source_id,
                    &doc_id,
                )
                .await
                {
                    Ok(()) => deleted += 1,
                    Err(error) => {
                        failed += 1;
                        log::warn!(
                            "Failed to delete document from RAG store: workspace_id={} source_id={} doc_id={} error={}",
                            workspace_id,
                            source_id,
                            doc_id,
                            error
                        );
                    }
                }
            }

            for payload in upsert_payloads {
                match ingest_document_payload_into_rag(
                    &rag_state,
                    &tenant_id,
                    &workspace_id,
                    &source_id,
                    &payload,
                )
                .await
                {
                    Ok(chunks) if chunks > 0 => {
                        upserted += 1;
                        chunk_count += chunks;
                    }
                    Ok(_) => {
                        skipped_non_text += 1;
                    }
                    Err(error) => {
                        if error.contains("unsupported non-text content type")
                            || error.contains("no text content")
                        {
                            skipped_non_text += 1;
                        } else {
                            failed += 1;
                            log::warn!(
                                "Failed to ingest document into RAG store: workspace_id={} source_id={} doc_id={} error={}",
                                workspace_id,
                                source_id,
                                payload.id,
                                error
                            );
                        }
                    }
                }
            }

            let next_cursor = deltas
                .iter()
                .filter_map(|delta| delta.modified_at.clone())
                .max();

            Ok::<_, warp::Rejection>(warp::reply::with_status(
                warp::reply::json(&json!({
                    "status": "ok",
                    "workspace_id": workspace_id,
                    "extension_id": extension_id,
                    "source_id": source_id,
                    "tenant_id": tenant_id,
                    "deltas_total": deltas.len(),
                    "upserted_docs": upserted,
                    "deleted_docs": deleted,
                    "skipped_docs": skipped_non_text,
                    "failed_docs": failed,
                    "chunks_upserted": chunk_count,
                    "next_cursor": next_cursor,
                })),
                StatusCode::OK,
            ))
        }
    });

    list_extensions
        .or(list_marketplace_extensions)
        .or(install_extension)
        .or(uninstall_extension)
        .or(set_extension_config)
        .or(get_extension_config)
        .or(launch_extension)
        .or(sync_extension)
        .map(reply_into_response)
        .boxed()
}
