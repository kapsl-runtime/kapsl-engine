use base64::engine::general_purpose::{
    STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL_SAFE_NO_PAD,
};
use base64::Engine as _;
use clap::{ArgGroup, ArgMatches, FromArgMatches, Parser};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
#[cfg(test)]
use futures::StreamExt;
use infer_adapter::{default_request_adapter_registry, parse_inference_request_with_registry};
use kapsl_backends::BackendFactory;
use kapsl_core::loader::Manifest;
use kapsl_core::{
    AutoScaler, EngineKind, ModelInfo, ModelRegistry, ModelStatus, PackageLoader, ScalingPolicy,
};
use kapsl_engine_api::{
    BatchingPolicy, BinaryTensorPacket, Engine, EngineError, EngineHandle, EngineMetrics,
    EngineModelInfo, InferenceRequest, TensorDtype,
};
#[cfg(feature = "gpu-device-pool")]
use kapsl_engine_api::{ExternalDeviceMemory, ExternalDeviceMemoryReport};
use kapsl_hal::device::DeviceInfo;
use kapsl_ipc::{IpcServer, TcpServer};
use kapsl_llm::block_manager::{new_shared_allocator, SharedBlockAllocator};
use kapsl_llm::global_scheduler::{EngineHandle as KvEngineHandle, GlobalKvScheduler};
use kapsl_llm::llm_backend::LLMBackend;
use kapsl_llm::rag::{
    build_rag_prompt, CitationStyle, RagChunk, RagPromptConfig, WhitespaceTokenCounter,
};
use kapsl_monitor::middleware::MonitoringMiddleware;
use kapsl_rag::extension::{
    ConnectorRuntimeHandle, ExtensionManager, ExtensionRegistry, InstalledExtension,
};
use kapsl_rag::vector::SqliteVectorStore;
use kapsl_rag::{
    AccessControl, ConnectorClient, DocStore, EmbeddedChunk, FsDocStore, VectorQuery, VectorStore,
};
use kapsl_rag_sdk::protocol::{ConnectorRequestKind, ConnectorResponseKind, ConnectorResult};
use kapsl_rag_sdk::types::{DeltaOp, DocumentDelta, DocumentPayload, SourceDescriptor};
use kapsl_scheduler::{
    determine_priority, PoolStrategy, ReplicaPool, ReplicaScheduler,
    RequestMetadata as SchedulerRequestMetadata, Scheduler,
};
use kapsl_shm::memory::ShmManager;
use kapsl_shm::{SchedulerSnapshot, ShmServer};
use kapsl_transport::TransportServer;
use parking_lot::{Mutex, RwLock};
use prometheus::Registry;
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::future::Future;
use std::io::{BufRead, BufWriter, Cursor, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, System};
use tar::{Archive, Builder};
use tokio::sync::Mutex as AsyncMutex;
#[cfg(test)]
use warp::Filter;

mod app;
mod backend_bundle;
mod backend_manager;
mod features;
mod http;
mod llama_cpp_backend_pack;
mod llama_cpp_shared_pool;
mod onnx_backend_pack;
mod runtime;
mod serving_backend;

use app::*;
use backend_bundle::*;
use backend_manager::*;
use features::*;
use http::*;
use llama_cpp_backend_pack::*;
use onnx_backend_pack::*;
use runtime::*;
use serving_backend::*;

type DynError = Box<dyn std::error::Error + Send + Sync>;
#[cfg(test)]
mod tests;

#[tokio::main]
async fn main() -> Result<(), DynError> {
    let raw_argv: Vec<String> = std::env::args().collect();
    let Cli {
        command,
        run: _legacy_run_args,
    } = Cli::parse_from(&raw_argv);
    match command {
        Some(KapslCommand::Build(args)) => return execute_build_command(args),
        Some(KapslCommand::Bundle(args)) => return execute_bundle_command(args),
        Some(KapslCommand::BackendPlan(args)) => return execute_backend_plan_command(args),
        Some(KapslCommand::Backend(args)) => return execute_backend_command(args),
        Some(KapslCommand::Push(args)) => return execute_push_command(args),
        Some(KapslCommand::Pull(args)) => return execute_pull_command(args),
        Some(KapslCommand::Login(args)) => return execute_login_command(args),
        Some(KapslCommand::Extension(args)) => return execute_extension_command(args),
        Some(KapslCommand::Provider(args)) => return execute_provider_command(args),
        Some(KapslCommand::AddModel(args)) => return execute_add_model_command(args),
        Some(KapslCommand::List(args)) => return execute_list_command(args),
        Some(KapslCommand::RemoveModel(args)) => return execute_remove_model_command(args),
        Some(KapslCommand::Run(_)) | None => {}
    }

    let runtime_argv = runtime_argv_from_invocation(&raw_argv);
    let (args, matches) = parse_runtime_args_and_matches(&runtime_argv)?;
    let startup_started_at = Instant::now();
    let config = resolve_runtime_config(args, &matches)?;
    env_logger::init();
    config.validate_and_log()?;

    let device_info = Arc::new(DeviceInfo::probe());
    RuntimeBootstrap::new(config, device_info)
        .prepare()
        .await?
        .run(startup_started_at)
        .await
}
