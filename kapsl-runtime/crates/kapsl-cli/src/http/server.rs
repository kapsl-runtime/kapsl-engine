//! HTTP API composition and server lifecycle.

use super::*;

/// Immutable HTTP listener and persistent-state settings.
pub(crate) struct HttpServerConfig {
    pub(crate) bind_addr: IpAddr,
    pub(crate) port: u16,
    pub(crate) state_layout: RuntimeStateLayout,
    pub(crate) auth_state: Arc<RwLock<ApiAuthState>>,
    pub(crate) log_sensitive_ids: bool,
}

/// Long-lived runtime services exposed through HTTP protocol adapters.
pub(crate) struct HttpServerDependencies {
    pub(crate) registry: Arc<Registry>,
    pub(crate) model_runtime: Arc<ModelRuntime>,
    pub(crate) inference: Arc<InferenceService>,
    pub(crate) telemetry: Arc<ModelTelemetry>,
    pub(crate) runtime_samples: Arc<RwLock<RuntimeSamples>>,
    pub(crate) runtime_pressure_state: Arc<AtomicU8>,
    pub(crate) auto_scaler: Arc<RwLock<AutoScaler>>,
    pub(crate) resources: Arc<RuntimeResources>,
}

/// Bound HTTP address and the task serving it.
pub(crate) struct HttpServerHandle {
    bound_addr: std::net::SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl HttpServerHandle {
    pub(crate) fn bound_addr(&self) -> std::net::SocketAddr {
        self.bound_addr
    }

    pub(crate) fn abort(&self) {
        self.task.abort();
    }

    pub(crate) async fn wait(&mut self) -> Result<(), tokio::task::JoinError> {
        (&mut self.task).await
    }
}

/// Build every HTTP route, bind the listener, and start serving it.
///
/// Binding happens before the task is spawned, so readiness is represented by
/// the successful return of this function rather than a separate oneshot and
/// timeout protocol in the application entrypoint.
pub(crate) fn start_http_server(
    config: HttpServerConfig,
    dependencies: HttpServerDependencies,
) -> Result<HttpServerHandle, DynError> {
    let HttpServerConfig {
        bind_addr,
        port,
        state_layout,
        auth_state,
        log_sensitive_ids,
    } = config;
    let HttpServerDependencies {
        registry,
        model_runtime,
        inference,
        telemetry,
        runtime_samples,
        runtime_pressure_state,
        auto_scaler,
        resources,
    } = dependencies;

    let models = model_runtime.models().clone();
    let device_info = Arc::new(model_runtime.device_info().clone());
    let extensions_root = state_layout.extensions_root;
    let extensions_config_root = state_layout.extensions_config_root;
    fs::create_dir_all(&extensions_root)?;
    fs::create_dir_all(&extensions_config_root)?;
    let extension_manager = Arc::new(ExtensionManager::new(
        ExtensionRegistry::new(extensions_root),
        extensions_config_root,
    ));
    let running_connectors: Arc<
        AsyncMutex<HashMap<String, ConnectorClient<ConnectorRuntimeHandle>>>,
    > = Arc::new(AsyncMutex::new(HashMap::new()));

    let rag_docs_root = state_layout.rag_root.join("docs");
    let rag_vector_path = state_layout.rag_root.join("vectors.sqlite3");
    fs::create_dir_all(&rag_docs_root)?;
    let rag_state = RagRuntimeState {
        vector_store: Arc::new(SqliteVectorStore::open(&rag_vector_path)?),
        doc_store: FsDocStore::new(&rag_docs_root),
    };

    let metrics_route =
        build_metrics_route(registry.clone(), auth_state.clone(), resources.clone());
    let model_routes = build_model_routes(ModelRoutesConfig {
        model_runtime,
        inference: inference.clone(),
        telemetry,
        rag_state: rag_state.clone(),
        auto_scaler,
        log_sensitive_ids,
    });
    let openai_routes = build_openai_routes(OpenAiRoutesConfig {
        models: models.clone(),
        inference,
        log_sensitive_ids,
    });
    let system_routes = build_system_routes(
        models,
        device_info,
        runtime_samples,
        runtime_pressure_state,
        resources,
    );
    let engine_routes = build_engine_routes();
    let extension_routes =
        build_extension_routes(extension_manager, running_connectors, rag_state.clone());
    let rag_routes = build_rag_routes(rag_state);
    let auth_routes = build_auth_routes(auth_state.clone());
    let static_routes = build_static_routes();

    let reader_api_routes = model_routes
        .reader
        .or(system_routes)
        .or(engine_routes.reader)
        .or(rag_routes)
        .or(openai_routes)
        .map(reply_into_response);
    let reader_api_routes = api_auth_filter(ApiRole::Reader, ApiScope::Read, auth_state.clone())
        .and(reader_api_routes)
        .map(|response: warp::reply::Response| response);

    let writer_api_routes = extension_routes.map(reply_into_response);
    let writer_api_routes = api_auth_filter(ApiRole::Writer, ApiScope::Write, auth_state.clone())
        .and(writer_api_routes)
        .map(|response: warp::reply::Response| response);

    let admin_api_routes = engine_routes
        .admin
        .or(model_routes.admin)
        .or(auth_routes.admin)
        .map(reply_into_response);
    let admin_api_routes = api_auth_filter(ApiRole::Admin, ApiScope::Admin, auth_state.clone())
        .and(admin_api_routes)
        .map(|response: warp::reply::Response| response);

    let api_routes = reader_api_routes
        .or(writer_api_routes)
        .unify()
        .or(admin_api_routes)
        .unify()
        .or_else(map_api_auth_rejection);
    let routes = static_routes
        .or(metrics_route)
        .or(auth_routes.login)
        .or(api_routes);

    let bind = (bind_addr, port);
    let (bound_addr, server) = warp::serve(routes)
        .try_bind_ephemeral(bind)
        .map_err(|error| {
            format!(
                "Failed to bind HTTP API on http://{}:{}/api: {}",
                bind_addr, port, error
            )
        })?;
    log_http_endpoints(bound_addr, &auth_state);
    let task = tokio::spawn(server);

    Ok(HttpServerHandle { bound_addr, task })
}

fn log_http_endpoints(bound_addr: std::net::SocketAddr, auth_state: &RwLock<ApiAuthState>) {
    log::info!("🌐 Web UI available at http://{bound_addr}/");
    log::info!("📊 Metrics available at http://{bound_addr}/metrics");
    log::info!("🔌 API available at http://{bound_addr}/api/");
    if auth_state.read().auth_enabled() {
        log::info!(
            "   - API auth roles: reader={}, writer={}, admin={} (Authorization: Bearer <api-key>)",
            API_READER_TOKEN_ENV,
            API_WRITER_TOKEN_ENV,
            API_ADMIN_TOKEN_ENV
        );
    }
    log::info!("   - GET /api/models - List all models");
    log::info!("   - GET /api/models/:id - Get model details");
    log::info!("   - POST /api/models/:id/remove - Remove a model");
    log::info!(
        "   - POST /api/models/:id/stage      - Pre-load next model into CPU RAM (hot-swap phase 1)"
    );
    log::info!("   - GET  /api/models/:id/swap-status - Check if staging is complete");
    log::info!(
        "   - POST /api/models/:id/swap        - Activate staged weights via PCIe transfer (hot-swap phase 2)"
    );
    log::info!("   - POST /api/models/:id/infer - Tensor or base64 media inference");
    log::info!("   - GET /api/health - System health check");
    log::info!("   - GET /api/hardware - Hardware info");
    log::info!("   - GET /api/system/stats - Runtime process stats (RSS/GPU util)");
    log::info!("   - POST /api/auth/login - Validate token and return effective access (public)");
    log::info!("   - POST /api/engine/package - Create a .aimod package");
    log::info!(
        "   - POST /api/engine/push - Push .aimod to remote backend (default: {})",
        DEFAULT_REMOTE_URL
    );
    log::info!("   - POST /api/engine/pull - Pull .aimod from remote backend");
    log::info!("   - GET /api/extensions - List extensions");
    log::info!(
        "   - GET /api/extensions/marketplace?q=... - Search marketplace extensions (default: {})",
        EXTENSION_MARKETPLACE_URL
    );
    log::info!("   - POST /api/extensions/install - Install extension");
    log::info!("   - POST /api/extensions/:id/uninstall - Uninstall extension");
    log::info!("   - POST /api/extensions/:id/config - Set extension config");
    log::info!("   - GET /api/extensions/:id/config?workspace_id=... - Get extension config");
    log::info!("   - POST /api/extensions/:id/launch - Launch connector");
    log::info!("   - POST /api/extensions/:id/sync - Sync connector docs into local RAG index");
    log::info!("   - GET /api/auth/roles - Read role token config (admin)");
    log::info!("   - POST /api/auth/roles - Update role token config (admin)");
    log::info!("   - GET /api/auth/access/status - Access control summary (admin)");
    log::info!("   - GET /api/auth/access/roles - Role summaries (admin)");
    log::info!("   - GET /api/auth/access/users - List users (admin)");
    log::info!("   - POST /api/auth/access/users - Create user (admin)");
    log::info!("   - PATCH /api/auth/access/users/:id - Update user role/status (admin)");
    log::info!("   - GET /api/auth/access/keys?user_id=... - List API keys (admin)");
    log::info!("   - POST /api/auth/access/users/:id/keys - Create API key (admin)");
    log::info!("   - POST /api/auth/access/keys/:id/revoke - Revoke API key (admin)");
    log::info!("   - POST /api/rag/query - Query indexed RAG chunks\n");
}
