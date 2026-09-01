#include <stdint.h>

#ifndef KAPSL_ONNX_PROFILE
#error "KAPSL_ONNX_PROFILE must be 1 (CPU), 2 (CUDA 12), or 3 (TensorRT 10)"
#endif

#if KAPSL_ONNX_PROFILE < 1 || KAPSL_ONNX_PROFILE > 3
#error "KAPSL_ONNX_PROFILE is outside the version-1 ABI"
#endif

#if defined(_WIN32)
#define KAPSL_EXPORT __declspec(dllexport)
#else
#define KAPSL_EXPORT __attribute__((visibility("default")))
#endif

struct kapsl_onnx_backend_pack_v1 {
  uint32_t magic;
  uint32_t struct_size;
  uint32_t runtime_abi;
  uint32_t profile;
};

KAPSL_EXPORT const struct kapsl_onnx_backend_pack_v1 *
kapsl_onnx_backend_pack_v1(void) {
  static const struct kapsl_onnx_backend_pack_v1 descriptor = {
      0x4b4f4e58u,
      (uint32_t)sizeof(struct kapsl_onnx_backend_pack_v1),
      1u,
      (uint32_t)KAPSL_ONNX_PROFILE,
  };
  return &descriptor;
}
