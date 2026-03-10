use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use bytes::Bytes;
use kapsl_rag_sdk::protocol::{ConnectorRequest, ConnectorRequestKind, ConnectorResponse};
use wasmtime::{Engine, Linker, Module, Store};
use wasmtime_wasi::preview2::pipe::{MemoryInputPipe, MemoryOutputPipe};
use wasmtime_wasi::preview2::preview1::{
    add_to_linker_sync, WasiPreview1Adapter, WasiPreview1View,
};
use wasmtime_wasi::preview2::{DirPerms, FilePerms, Table, WasiCtx, WasiCtxBuilder, WasiView};

#[derive(thiserror::Error, Debug)]
pub enum RuntimeError {
    #[error("io error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("connector exited")]
    ConnectorExited,
    #[error("wasm error: {0}")]
    Wasm(String),
}

impl From<io::Error> for RuntimeError {
    fn from(err: io::Error) -> Self {
        RuntimeError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for RuntimeError {
    fn from(err: serde_json::Error) -> Self {
        RuntimeError::Serialization(err.to_string())
    }
}

pub trait ConnectorRuntime {
    fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError>;
    fn close(&mut self) -> Result<(), RuntimeError>;
}

struct WasiState {
    table: Table,
    ctx: WasiCtx,
    adapter: WasiPreview1Adapter,
}

impl WasiState {
    fn new(ctx: WasiCtx) -> Self {
        Self {
            table: Table::new(),
            ctx,
            adapter: WasiPreview1Adapter::new(),
        }
    }
}

impl WasiView for WasiState {
    fn table(&self) -> &Table {
        &self.table
    }

    fn table_mut(&mut self) -> &mut Table {
        &mut self.table
    }

    fn ctx(&self) -> &WasiCtx {
        &self.ctx
    }

    fn ctx_mut(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
}

impl WasiPreview1View for WasiState {
    fn adapter(&self) -> &WasiPreview1Adapter {
        &self.adapter
    }

    fn adapter_mut(&mut self) -> &mut WasiPreview1Adapter {
        &mut self.adapter
    }
}

#[derive(Debug, Clone, Default)]
pub struct WasiPermissions {
    pub preopen_dirs: Vec<PreopenDir>,
    pub env: HashMap<String, String>,
}

impl WasiPermissions {
    pub fn allow_dir(
        mut self,
        host_path: impl Into<PathBuf>,
        guest_path: impl Into<String>,
        read_only: bool,
    ) -> Self {
        self.preopen_dirs.push(PreopenDir {
            host_path: host_path.into(),
            guest_path: guest_path.into(),
            read_only,
        });
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct PreopenDir {
    pub host_path: PathBuf,
    pub guest_path: String,
    pub read_only: bool,
}

pub struct ConnectorClient<R: ConnectorRuntime> {
    runtime: R,
    next_id: u64,
}

impl<R: ConnectorRuntime> ConnectorClient<R> {
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            next_id: 1,
        }
    }

    pub fn request(
        &mut self,
        kind: ConnectorRequestKind,
    ) -> Result<ConnectorResponse, RuntimeError> {
        let id = format!("req-{}", self.next_id);
        self.next_id += 1;
        let request = ConnectorRequest { id, kind };
        self.runtime.send(request)
    }

    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        self.runtime.close()
    }
}

pub struct SidecarConnectorRuntime {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl SidecarConnectorRuntime {
    pub fn spawn(path: &Path) -> Result<Self, RuntimeError> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().ok_or(RuntimeError::ConnectorExited)?;
        let stdout = child.stdout.take().ok_or(RuntimeError::ConnectorExited)?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }
}

impl ConnectorRuntime for SidecarConnectorRuntime {
    fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError> {
        let json = serde_json::to_string(&request)?;
        self.stdin.write_all(json.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let mut line = String::new();
        let bytes = self.stdout.read_line(&mut line)?;
        if bytes == 0 {
            return Err(RuntimeError::ConnectorExited);
        }
        let response = serde_json::from_str(&line)?;
        Ok(response)
    }

    fn close(&mut self) -> Result<(), RuntimeError> {
        let _ = self.child.id();
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }
}

pub struct WasmConnectorRuntime {
    engine: Engine,
    module: Module,
    permissions: WasiPermissions,
}

impl WasmConnectorRuntime {
    pub fn spawn(path: &Path) -> Result<Self, RuntimeError> {
        Self::spawn_with_permissions(path, WasiPermissions::default())
    }

    pub fn spawn_with_permissions(
        path: &Path,
        permissions: WasiPermissions,
    ) -> Result<Self, RuntimeError> {
        let engine = Engine::default();
        let module =
            Module::from_file(&engine, path).map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        Ok(Self {
            engine,
            module,
            permissions,
        })
    }

    fn run_once(&self, input: &str) -> Result<String, RuntimeError> {
        // Run the WASM connector to completion for this request. This keeps the
        // implementation simple and sandboxed, at the cost of per-request startup.
        let mut linker = Linker::new(&self.engine);
        add_to_linker_sync(&mut linker).map_err(|e| RuntimeError::Wasm(e.to_string()))?;

        let stdin = MemoryInputPipe::new(Bytes::from(input.as_bytes().to_vec()));
        let stdout = MemoryOutputPipe::new(4 * 1024 * 1024);
        let stderr = MemoryOutputPipe::new(256 * 1024);

        let mut builder = WasiCtxBuilder::new();
        let _ = builder.stdin(stdin);
        let _ = builder.stdout(stdout.clone());
        let _ = builder.stderr(stderr.clone());

        for (key, value) in &self.permissions.env {
            validate_env_kv(key, value)?;
            let _ = builder.env(key, value);
        }

        for dir in &self.permissions.preopen_dirs {
            validate_guest_path(&dir.guest_path)?;
            let cap_dir =
                cap_std::fs::Dir::open_ambient_dir(&dir.host_path, cap_std::ambient_authority())
                    .map_err(|e| RuntimeError::Wasm(e.to_string()))?;
            let (dir_perms, file_perms) = perms_for(dir.read_only);
            let _ = builder.preopened_dir(cap_dir, dir_perms, file_perms, &dir.guest_path);
        }

        let wasi = builder.build();
        let mut store = Store::new(&self.engine, WasiState::new(wasi));
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;

        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        start
            .call(&mut store, ())
            .map_err(|e| RuntimeError::Wasm(e.to_string()))?;

        let output = stdout.contents();
        let output =
            String::from_utf8(output.to_vec()).map_err(|e| RuntimeError::Wasm(e.to_string()))?;
        Ok(output)
    }
}

impl ConnectorRuntime for WasmConnectorRuntime {
    fn send(&mut self, request: ConnectorRequest) -> Result<ConnectorResponse, RuntimeError> {
        let payload = serde_json::to_string(&request)?;
        let output = self.run_once(&format!("{payload}\n"))?;
        let mut last_line = None;
        for line in output.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                last_line = Some(trimmed.to_string());
            }
        }
        let last_line = last_line.ok_or(RuntimeError::ConnectorExited)?;
        let response = serde_json::from_str(&last_line)?;
        Ok(response)
    }

    fn close(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }
}

fn validate_env_kv(key: &str, value: &str) -> Result<(), RuntimeError> {
    if key.is_empty() {
        return Err(RuntimeError::Wasm("env key cannot be empty".to_string()));
    }
    if key.contains('\0') || value.contains('\0') {
        return Err(RuntimeError::Wasm(
            "env key/value cannot contain NUL".to_string(),
        ));
    }
    Ok(())
}

fn validate_guest_path(path: &str) -> Result<(), RuntimeError> {
    if path.is_empty() || !path.starts_with('/') {
        return Err(RuntimeError::Wasm(
            "preopened guest path must be absolute".to_string(),
        ));
    }
    if path.contains('\0') {
        return Err(RuntimeError::Wasm(
            "preopened guest path cannot contain NUL".to_string(),
        ));
    }
    Ok(())
}

fn perms_for(read_only: bool) -> (DirPerms, FilePerms) {
    let dir_perms = if read_only {
        DirPerms::READ
    } else {
        DirPerms::READ | DirPerms::MUTATE
    };
    let file_perms = if read_only {
        FilePerms::READ
    } else {
        FilePerms::READ | FilePerms::WRITE
    };
    (dir_perms, file_perms)
}
