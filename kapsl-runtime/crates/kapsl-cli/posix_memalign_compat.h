// Windows posix_memalign compatibility shim
// This file provides a POSIX-compatible posix_memalign for Windows

#ifndef WINDOWS_POSIX_MEMALIGN_H
#define WINDOWS_POSIX_MEMALIGN_H

#ifdef _WIN32

#include <malloc.h>
#include <errno.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// posix_memalign compatibility for Windows
// Allocates size bytes aligned to alignment boundary
int posix_memalign(void** memptr, size_t alignment, size_t size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // _WIN32

#endif  // WINDOWS_POSIX_MEMALIGN_H

