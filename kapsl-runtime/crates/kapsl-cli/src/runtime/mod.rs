use super::*;
use futures::stream;
use warp::Reply;

#[cfg(unix)]
use std::os::unix::net::UnixStream;

pub(crate) mod autoscaler;
pub(crate) mod config;
#[cfg(all(feature = "gpu-device-pool", target_os = "linux"))]
pub(crate) mod cuda_ipc;
#[cfg(any(feature = "gpu-device-pool", test))]
mod device_budget;
#[cfg(feature = "gpu-device-pool")]
pub(crate) mod device_memory;
mod host_memory;
pub(crate) mod inference_service;
#[cfg(unix)]
pub(crate) mod kv_control;
pub(crate) mod kv_runtime;
pub(crate) mod managed_vllm;
pub(crate) mod managed_vllm_bridge;
pub(crate) mod memory;
pub(crate) mod model;
pub(crate) mod model_manager;
pub(crate) mod monitor;
pub(crate) mod priority_arbiter;
pub(crate) mod resources;
pub(crate) mod shared_kv;
pub(crate) mod supervisor;
pub(crate) mod support;
pub(crate) mod transport;
pub(crate) mod tuning;
pub(crate) mod worker;

pub(crate) use autoscaler::*;
pub(crate) use config::*;
#[cfg(all(feature = "gpu-device-pool", target_os = "linux"))]
pub(crate) use cuda_ipc::*;
#[cfg(feature = "gpu-device-pool")]
pub(crate) use device_memory::*;
pub(crate) use inference_service::*;
#[cfg(unix)]
pub(crate) use kv_control::*;
pub(crate) use kv_runtime::*;
pub(crate) use managed_vllm::*;
pub(crate) use memory::*;
#[cfg(feature = "gpu-device-pool")]
pub(crate) use model::load_plan::device_memory_bootstrap_plan;
pub(crate) use model::*;
pub(crate) use model_manager::*;
pub(crate) use monitor::*;
pub(crate) use priority_arbiter::*;
pub(crate) use resources::*;
pub(crate) use shared_kv::*;
pub(crate) use supervisor::*;
pub(crate) use support::*;
pub(crate) use transport::*;
pub(crate) use tuning::*;
pub(crate) use worker::*;
