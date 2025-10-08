#pragma once

#include "cortex_llm.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// GGUF file format constants
#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3

// GGUF data types
typedef enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

// GGUF tensor types
typedef enum {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8 = 16,
    GGML_TYPE_I16 = 17,
    GGML_TYPE_I32 = 18,
    GGML_TYPE_COUNT = 19,
} ggml_type;

// GGUF file header structure
typedef struct {
    char magic[4];          // "GGUF"
    uint32_t version;       // version number
    uint64_t n_tensors;     // number of tensors
    uint64_t n_kv;          // number of key-value pairs
} gguf_header;

// GGUF key-value pair structure
typedef struct {
    uint64_t key_len;       // key length
    char *key;              // key string
    gguf_type type;         // value type
    union {
        uint8_t uint8_val;
        int8_t int8_val;
        uint16_t uint16_val;
        int16_t int16_val;
        uint32_t uint32_val;
        int32_t int32_val;
        float float32_val;
        bool bool_val;
        char *string_val;
        struct {
            gguf_type type;
            uint64_t n;
            void *data;
        } array_val;
        uint64_t uint64_val;
        int64_t int64_val;
        double float64_val;
    } value;
} gguf_kv;

// GGUF tensor information structure
typedef struct {
    uint64_t name_len;      // name length
    char *name;             // tensor name
    uint32_t n_dims;        // number of dimensions
    uint64_t shape[4];      // shape
    ggml_type type;         // data type
    uint64_t offset;        // data offset
} gguf_tensor_info;

// GGUF loader structure
typedef struct {
    FILE *file;             // file pointer
    gguf_header header;     // file header
    gguf_kv *kv_pairs;      // key-value pair array
    gguf_tensor_info *tensors; // tensor information array
    void *tensor_data;      // tensor data
    size_t tensor_data_size; // tensor data size
} gguf_loader;

// Function declarations
cortex_error load_gguf_file(const char *filename, gguf_loader *loader);
void free_gguf_loader(gguf_loader *loader);
gguf_kv* find_kv(gguf_loader *loader, const char *key);
gguf_tensor_info* find_tensor(gguf_loader *loader, const char *name);
size_t ggml_type_size(ggml_type type);
size_t ggml_blck_size(ggml_type type);
