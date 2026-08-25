#ifndef KAPSL_LLAMA_CPP_BACKEND_H
#define KAPSL_LLAMA_CPP_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KAPSL_LLAMA_CPP_ABI_VERSION 1u
#define KAPSL_LLAMA_CPP_ENTRYPOINT_MAGIC 0x4b4c4c4du
#define KAPSL_LLAMA_CPP_WIRE_FORMAT_JSON_V1 1u

#define KAPSL_STATUS_OK 0
#define KAPSL_STATUS_INVALID_ARGUMENT 1
#define KAPSL_STATUS_INCOMPATIBLE_ABI 2
#define KAPSL_STATUS_UNSUPPORTED 3
#define KAPSL_STATUS_BACKEND_ERROR 4
#define KAPSL_STATUS_CANCELLED 5
#define KAPSL_STATUS_PANIC 6

#define KAPSL_LLAMA_CAP_CPU (UINT64_C(1) << 0)
#define KAPSL_LLAMA_CAP_CUDA (UINT64_C(1) << 1)
#define KAPSL_LLAMA_CAP_NATIVE_KV (UINT64_C(1) << 2)
#define KAPSL_LLAMA_CAP_SHARED_POOL (UINT64_C(1) << 3)
#define KAPSL_LLAMA_CAP_STREAMING (UINT64_C(1) << 4)
#define KAPSL_LLAMA_CAP_CANCELLATION (UINT64_C(1) << 5)
#define KAPSL_LLAMA_CAP_MEMORY_REPORTING (UINT64_C(1) << 6)

#define KAPSL_LLAMA_PROFILE_CPU 1u
#define KAPSL_LLAMA_PROFILE_CUDA12 2u

#define KAPSL_LOG_ERROR 1u
#define KAPSL_LOG_WARN 2u
#define KAPSL_LOG_INFO 3u
#define KAPSL_LOG_DEBUG 4u
#define KAPSL_LOG_TRACE 5u

typedef struct kapsl_slice {
    const uint8_t *ptr;
    size_t len;
} kapsl_slice;

typedef struct kapsl_owned_buffer {
    uint8_t *ptr;
    size_t len;
    size_t capacity;
} kapsl_owned_buffer;

typedef void (*kapsl_log_fn)(void *user_data, uint32_t level, kapsl_slice message);
typedef uint32_t (*kapsl_request_cancelled_fn)(void *user_data, uint64_t request_id);

typedef struct kapsl_shared_pool_geometry_v1 {
    uint32_t struct_size;
    uint32_t device_id;
    uint64_t requested_blocks;
    uint32_t block_size_tokens;
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t key_head_dim;
    uint32_t value_head_dim;
    uint32_t element_bytes;
    uint32_t max_sequences;
    uint32_t max_blocks_per_sequence;
    uint64_t model_fingerprint;
} kapsl_shared_pool_geometry_v1;

typedef uint32_t (*kapsl_pool_reserve_fn)(
    void *pool_context,
    uint64_t session_id,
    uint32_t tokens_needed,
    uint32_t **block_table_device_out,
    uint32_t *blocks_out);
typedef uint32_t (*kapsl_pool_commit_sequences_fn)(
    void *pool_context,
    uint32_t **block_table_device_out);
typedef void (*kapsl_pool_release_fn)(void *pool_context, uint64_t session_id);
typedef uint32_t (*kapsl_pool_touch_fn)(void *pool_context, uint64_t session_id);

typedef struct kapsl_shared_pool_descriptor_v1 {
    uint32_t struct_size;
    void *pool_context;
    void *device_base;
    uint64_t addressable_blocks;
    uint32_t *block_table_device;
    uint32_t block_table_layer_stride;
    uint32_t block_table_sequence_stride;
    uint32_t sequence_slots;
    kapsl_pool_reserve_fn reserve;
    kapsl_pool_reserve_fn reserve_sequence;
    kapsl_pool_commit_sequences_fn commit_sequences;
    kapsl_pool_release_fn release;
    kapsl_pool_touch_fn touch;
} kapsl_shared_pool_descriptor_v1;

typedef int32_t (*kapsl_create_shared_pool_fn)(
    void *user_data,
    const kapsl_shared_pool_geometry_v1 *geometry,
    kapsl_shared_pool_descriptor_v1 *descriptor_out,
    kapsl_owned_buffer *error_out);
typedef void (*kapsl_destroy_shared_pool_fn)(void *user_data, void *pool_context);
typedef uint64_t (*kapsl_shared_pool_bytes_fn)(void *user_data, void *pool_context);

typedef struct kapsl_llama_host_callbacks_v1 {
    uint32_t struct_size;
    void *user_data;
    kapsl_log_fn log;
    kapsl_create_shared_pool_fn create_shared_pool;
    kapsl_destroy_shared_pool_fn destroy_shared_pool;
    kapsl_shared_pool_bytes_fn shared_pool_bytes;
} kapsl_llama_host_callbacks_v1;

typedef struct kapsl_llama_config_v1 {
    uint32_t struct_size;
    uint32_t profile;
    uint32_t device_id;
    uint32_t model_id;
    uint32_t replica_id;
    uint32_t require_shared_pool;
    /* The callback table and context remain live until shutdown returns. */
    const kapsl_llama_host_callbacks_v1 *host;
} kapsl_llama_config_v1;

typedef struct kapsl_llama_request_v1 {
    uint32_t struct_size;
    uint32_t wire_format;
    uint64_t request_id;
    kapsl_slice request_json;
    void *cancellation_context;
    kapsl_request_cancelled_fn is_cancelled;
} kapsl_llama_request_v1;

typedef int32_t (*kapsl_llama_stream_chunk_fn)(
    void *user_data,
    uint64_t request_id,
    kapsl_slice packet_json);

typedef struct kapsl_llama_cpp_api_v1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t wire_format;
    uint64_t capabilities;
    int32_t (*initialize)(const kapsl_llama_config_v1 *, void **, kapsl_owned_buffer *);
    int32_t (*planned_memory)(void *, kapsl_slice, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*load_model)(void *, kapsl_slice, kapsl_owned_buffer *);
    int32_t (*planned_request_memory)(void *, const kapsl_llama_request_v1 *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*infer)(void *, const kapsl_llama_request_v1 *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*infer_stream)(void *, const kapsl_llama_request_v1 *, void *, kapsl_llama_stream_chunk_fn, kapsl_owned_buffer *);
    int32_t (*cancel)(void *, uint64_t);
    int32_t (*actual_memory)(void *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*metrics)(void *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*model_info)(void *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*kv_capabilities)(void *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*kv_topology)(void *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*batching_policy)(void *, kapsl_owned_buffer *, kapsl_owned_buffer *);
    int32_t (*health_check)(void *, kapsl_owned_buffer *);
    void (*unload)(void *);
    void (*shutdown)(void *);
    void (*free_buffer)(kapsl_owned_buffer);
} kapsl_llama_cpp_api_v1;

const kapsl_llama_cpp_api_v1 *kapsl_llama_cpp_backend_v1(void);

/* Packaging-time marker; runtime authority remains the signed manifest plus
 * the capabilities returned by kapsl_llama_cpp_backend_v1(). */
extern const char KAPSL_LLAMA_CPP_KV_MODE_V1[];

#ifdef __cplusplus
}
#endif

#endif
