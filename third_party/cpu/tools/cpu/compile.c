/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// kernel function pointer type: args... + gridX, gridY, gridZ, numGridsX, numGridsY, numGridsZ
typedef void (*kernel_func_t)({kernel_fn_arg_types});

// globals
static void *{kernel_name}_lib = NULL;
static kernel_func_t {kernel_name}_func = NULL;

// Embedded .so binary data
unsigned char {kernel_name}_bin[{bin_size}] = {{ {bin_data} }};

void unload_{kernel_name}(void) {{
    if ({kernel_name}_lib) {{
        dlclose({kernel_name}_lib);
        {kernel_name}_lib = NULL;
        {kernel_name}_func = NULL;
    }}
}}

void load_{kernel_name}(void) {{
    // Write embedded .so to a temp file and dlopen it
    char tmppath[] = "/tmp/triton_aot_XXXXXX.so";
    int fd = mkstemps(tmppath, 3);
    if (fd < 0) {{
        fprintf(stderr, "Triton AOT Error: failed to create temp file\\n");
        exit(1);
    }}
    ssize_t written = write(fd, {kernel_name}_bin, {bin_size});
    close(fd);
    if (written != {bin_size}) {{
        fprintf(stderr, "Triton AOT Error: failed to write kernel binary\\n");
        unlink(tmppath);
        exit(1);
    }}

    {kernel_name}_lib = dlopen(tmppath, RTLD_NOW | RTLD_LOCAL);
    unlink(tmppath);  // safe to unlink after dlopen
    if (!{kernel_name}_lib) {{
        fprintf(stderr, "Triton AOT Error: dlopen failed: %s\\n", dlerror());
        exit(1);
    }}

    {kernel_name}_func = (kernel_func_t)dlsym({kernel_name}_lib, "{triton_kernel_name}");
    if (!{kernel_name}_func) {{
        fprintf(stderr, "Triton AOT Error: dlsym failed for {triton_kernel_name}: %s\\n", dlerror());
        dlclose({kernel_name}_lib);
        {kernel_name}_lib = NULL;
        exit(1);
    }}
}}

/*
{kernel_docstring}
*/
int {kernel_name}({signature}) {{
    if ({kernel_name}_func == NULL)
        load_{kernel_name}();

    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    unsigned int N = gX * gY * gZ;

    if (N == 0)
        return 0;

    if (N == 1) {{
        {kernel_name}_func({kernel_fn_args_list}{kernel_fn_args_comma}0, 0, 0, 1, 1, 1);
        return 0;
    }}

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (max_threads > 1) {{
        #pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < N; ++i) {{
            unsigned int z = i / (gX * gY);
            unsigned int rem = i % (gX * gY);
            unsigned int y = rem / gX;
            unsigned int x = rem % gX;
            {kernel_name}_func({kernel_fn_args_list}{kernel_fn_args_comma}x, y, z, gX, gY, gZ);
        }}
    }} else
#endif
    {{
        for (unsigned int z = 0; z < gZ; ++z) {{
            for (unsigned int y = 0; y < gY; ++y) {{
                for (unsigned int x = 0; x < gX; ++x) {{
                    {kernel_name}_func({kernel_fn_args_list}{kernel_fn_args_comma}x, y, z, gX, gY, gZ);
                }}
            }}
        }}
    }}

    return 0;
}}
