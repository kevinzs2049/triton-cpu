/* clang-format off */
#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
int{_placeholder} {kernel_name}({signature});
