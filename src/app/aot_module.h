#pragma once
#include <stdint.h>

#ifndef GRAPH_TCM_PATH
#error "GRAPH_TCM_PATH must be defined by CMake"
#endif

__asm__(".section .rdata,\"dr\"\n"
        ".global g_graphs_tcm_start\n"
        "g_graphs_tcm_start:\n"
    ".incbin \"" GRAPH_TCM_PATH "\"\n"
        ".global g_graphs_tcm_end\n"
        "g_graphs_tcm_end:\n");

// 2. 声明汇编中定义的全局符号
#ifdef __cplusplus
extern "C"
{
#endif
    extern const uint8_t g_graphs_tcm_start[];
    extern const uint8_t g_graphs_tcm_end[];
#ifdef __cplusplus
}
#endif