/*
 * Compatibility shim: ort-sys prebuilts are compiled against glibc 2.38+ which
 * introduced __isoc23_* versioned aliases for the C23 string conversion
 * functions.  Clusters running older glibc lack these symbols, causing a link
 * failure.  The C23 versions have identical ABI to their predecessors, so
 * simple forwarders are correct.
 */
#include <stdlib.h>

long long __isoc23_strtoll(const char *s, char **e, int b)   { return strtoll(s, e, b); }
unsigned long long __isoc23_strtoull(const char *s, char **e, int b) { return strtoull(s, e, b); }
long __isoc23_strtol(const char *s, char **e, int b)         { return strtol(s, e, b); }
