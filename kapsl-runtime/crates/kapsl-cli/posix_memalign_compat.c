// Windows posix_memalign compatibility shim
// This provides a POSIX-compatible posix_memalign function for Windows
// that uses _aligned_malloc internally

#ifdef _WIN32

#include <malloc.h>
#include <errno.h>
#include <stddef.h>

// Provide posix_memalign for Windows
// Matches the POSIX signature and behavior
int posix_memalign(void **memptr, size_t alignment, size_t size) {
    // Validate alignment
    if (alignment < sizeof(void*)) {
        return EINVAL;
    }
    
    // Check if alignment is a power of 2
    if ((alignment & (alignment - 1)) != 0) {
        return EINVAL;
    }
    
    // Allocate aligned memory using Windows API
    void *ptr = _aligned_malloc(size, alignment);
    if (ptr == NULL) {
        return ENOMEM;
    }
    
    *memptr = ptr;
    return 0;
}

#endif  // _WIN32
