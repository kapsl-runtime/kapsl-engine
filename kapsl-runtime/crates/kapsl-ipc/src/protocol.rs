use kapsl_transport::{RequestMetadata, ResponseMetadata};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct RequestHeader {
    pub model_id: u32,
    pub op_code: u32, // 1=infer, 2=stream, 3=metrics, 4=hybrid_infer
    pub payload_size: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct ResponseHeader {
    pub status: u32, // 0=OK, 1=ERR
    pub payload_size: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HybridRequest {
    pub metadata: RequestMetadata,
    pub shm_offset: u64,
    pub shm_size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HybridResponse {
    pub metadata: ResponseMetadata,
    pub shm_offset: u64,
    pub shm_size: u64,
}

pub const OP_INFER: u32 = 1;
pub const OP_INFER_STREAM: u32 = 2;
pub const OP_METRICS: u32 = 3;
pub const OP_HYBRID_INFER: u32 = 4;

pub const STATUS_OK: u32 = 0;
pub const STATUS_ERR: u32 = 1;
pub const STATUS_STREAM_CHUNK: u32 = 2;
pub const STATUS_STREAM_END: u32 = 3;
