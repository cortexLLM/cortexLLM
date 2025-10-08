#include "cortex_llm.h"
#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>


// Helper function: read integer
static bool read_uint32(FILE *file, uint32_t *value) {
    return fread(value, sizeof(uint32_t), 1, file) == 1;
}

static bool read_uint64(FILE *file, uint64_t *value) {
    return fread(value, sizeof(uint64_t), 1, file) == 1;
}

// Helper function: read string
static char* read_string(FILE *file, uint64_t len) {
    char *str = malloc(len + 1);
    if (!str) return NULL;
    
    if (fread(str, 1, len, file) != len) {
        free(str);
        return NULL;
    }
    str[len] = '\0';
    return str;
}

// Helper function: get GGML type size
size_t ggml_type_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 66;
        case GGML_TYPE_Q3_K: return 80;
        case GGML_TYPE_Q4_K: return 96;
        case GGML_TYPE_Q5_K: return 112;
        case GGML_TYPE_Q6_K: return 128;
        case GGML_TYPE_Q8_K: return 144;
        case GGML_TYPE_I8: return 1;
        case GGML_TYPE_I16: return 2;
        case GGML_TYPE_I32: return 4;
        default: return 0;
    }
}

// Helper function: get GGML type block size
size_t ggml_blck_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            return 32;
        default:
            return 1;
    }
}

// Load GGUF file
cortex_error load_gguf_file(const char *filename, gguf_loader *loader) {
    loader->file = fopen(filename, "rb");
    if (!loader->file) {
        return CORTEX_ERROR_FILE_LOAD;
    }
    
    // Read file header
    if (fread(&loader->header.magic, 4, 1, loader->file) != 1) {
        fclose(loader->file);
        return CORTEX_ERROR_FILE_LOAD;
    }
    
    if (memcmp(loader->header.magic, GGUF_MAGIC, 4) != 0) {
        fclose(loader->file);
        return CORTEX_ERROR_INVALID_MODEL;
    }
    
    if (!read_uint32(loader->file, &loader->header.version) ||
        !read_uint64(loader->file, &loader->header.n_tensors) ||
        !read_uint64(loader->file, &loader->header.n_kv)) {
        fclose(loader->file);
        return CORTEX_ERROR_FILE_LOAD;
    }
    
    // Allocate key-value pair array
    loader->kv_pairs = malloc(loader->header.n_kv * sizeof(gguf_kv));
    if (!loader->kv_pairs) {
        fclose(loader->file);
        return CORTEX_ERROR_MEMORY;
    }
    
    // Read key-value pairs
    for (uint64_t i = 0; i < loader->header.n_kv; i++) {
        gguf_kv *kv = &loader->kv_pairs[i];
        
        if (!read_uint64(loader->file, &kv->key_len)) {
            fclose(loader->file);
            free(loader->kv_pairs);
            return CORTEX_ERROR_FILE_LOAD;
        }
        
        kv->key = read_string(loader->file, kv->key_len);
        if (!kv->key) {
            fclose(loader->file);
            free(loader->kv_pairs);
            return CORTEX_ERROR_FILE_LOAD;
        }
        
        if (fread(&kv->type, sizeof(gguf_type), 1, loader->file) != 1) {
            fclose(loader->file);
            free(kv->key);
            free(loader->kv_pairs);
            return CORTEX_ERROR_FILE_LOAD;
        }
        
        // Read value
        switch (kv->type) {
            case GGUF_TYPE_UINT8:
                if (fread(&kv->value.uint8_val, 1, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_INT8:
                if (fread(&kv->value.int8_val, 1, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_UINT16:
                if (fread(&kv->value.uint16_val, 2, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_INT16:
                if (fread(&kv->value.int16_val, 2, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_UINT32:
                if (fread(&kv->value.uint32_val, 4, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_INT32:
                if (fread(&kv->value.int32_val, 4, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_FLOAT32:
                if (fread(&kv->value.float32_val, 4, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_BOOL:
                if (fread(&kv->value.bool_val, 1, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_STRING: {
                uint64_t str_len;
                if (!read_uint64(loader->file, &str_len)) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                kv->value.string_val = read_string(loader->file, str_len);
                if (!kv->value.string_val) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            }
            case GGUF_TYPE_UINT64:
                if (fread(&kv->value.uint64_val, 8, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_INT64:
                if (fread(&kv->value.int64_val, 8, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_FLOAT64:
                if (fread(&kv->value.float64_val, 8, 1, loader->file) != 1) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                break;
            case GGUF_TYPE_ARRAY: {
                // Read array type
                gguf_type array_type;
                uint64_t array_len;
                if (fread(&array_type, sizeof(gguf_type), 1, loader->file) != 1 ||
                    !read_uint64(loader->file, &array_len)) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_FILE_LOAD;
                }
                
                kv->value.array_val.type = array_type;
                kv->value.array_val.n = array_len;
                
                // Allocate array data memory
                size_t element_size = 0;
                switch (array_type) {
                    case GGUF_TYPE_UINT8: element_size = 1; break;
                    case GGUF_TYPE_INT8: element_size = 1; break;
                    case GGUF_TYPE_UINT16: element_size = 2; break;
                    case GGUF_TYPE_INT16: element_size = 2; break;
                    case GGUF_TYPE_UINT32: element_size = 4; break;
                    case GGUF_TYPE_INT32: element_size = 4; break;
                    case GGUF_TYPE_FLOAT32: element_size = 4; break;
                    case GGUF_TYPE_BOOL: element_size = 1; break;
                    case GGUF_TYPE_STRING: element_size = sizeof(char*); break;
                    default: element_size = 1; break;
                }
                
                kv->value.array_val.data = malloc(array_len * element_size);
                if (!kv->value.array_val.data) {
                    fclose(loader->file);
                    free(kv->key);
                    free(loader->kv_pairs);
                    return CORTEX_ERROR_MEMORY;
                }
                
                // Read array data
                if (array_type == GGUF_TYPE_STRING) {
                    // String arrays need special handling
                    char **str_array = (char**)kv->value.array_val.data;
                    for (uint64_t j = 0; j < array_len; j++) {
                        uint64_t str_len;
                        if (!read_uint64(loader->file, &str_len)) {
                            fclose(loader->file);
                            free(kv->key);
                            free(loader->kv_pairs);
                            return CORTEX_ERROR_FILE_LOAD;
                        }
                        str_array[j] = read_string(loader->file, str_len);
                        if (!str_array[j]) {
                            fclose(loader->file);
                            free(kv->key);
                            free(loader->kv_pairs);
                            return CORTEX_ERROR_FILE_LOAD;
                        }
                    }
                } else {
                    // Other types read directly
                    if (fread(kv->value.array_val.data, element_size, array_len, loader->file) != array_len) {
                        fclose(loader->file);
                        free(kv->key);
                        free(loader->kv_pairs);
                        return CORTEX_ERROR_FILE_LOAD;
                    }
                }
                break;
            }
            default:
                printf("Error: unknown key-value pair type %d\n", kv->type);
                fclose(loader->file);
                free(kv->key);
                free(loader->kv_pairs);
                return CORTEX_ERROR_INVALID_MODEL;
        }
    }
    
    // Allocate tensor information array
    loader->tensors = malloc(loader->header.n_tensors * sizeof(gguf_tensor_info));
    if (!loader->tensors) {
        fclose(loader->file);
        free(loader->kv_pairs);
        return CORTEX_ERROR_MEMORY;
    }
    
    // Read tensor information
    for (uint64_t i = 0; i < loader->header.n_tensors; i++) {
        gguf_tensor_info *tensor = &loader->tensors[i];
        
        if (!read_uint64(loader->file, &tensor->name_len)) {
            fclose(loader->file);
            free(loader->kv_pairs);
            free(loader->tensors);
            return CORTEX_ERROR_FILE_LOAD;
        }
        
        tensor->name = read_string(loader->file, tensor->name_len);
        if (!tensor->name) {
            fclose(loader->file);
            free(loader->kv_pairs);
            free(loader->tensors);
            return CORTEX_ERROR_FILE_LOAD;
        }
        
        if (!read_uint32(loader->file, &tensor->n_dims)) {
            fclose(loader->file);
            free(loader->kv_pairs);
            free(loader->tensors);
            free(tensor->name);
            return CORTEX_ERROR_FILE_LOAD;
        }
        
        for (uint32_t j = 0; j < tensor->n_dims; j++) {
            if (!read_uint64(loader->file, &tensor->shape[j])) {
                fclose(loader->file);
                free(loader->kv_pairs);
                free(loader->tensors);
                free(tensor->name);
                return CORTEX_ERROR_FILE_LOAD;
            }
        }
        
        if (fread(&tensor->type, sizeof(ggml_type), 1, loader->file) != 1 ||
            !read_uint64(loader->file, &tensor->offset)) {
            fclose(loader->file);
            free(loader->kv_pairs);
            free(loader->tensors);
            free(tensor->name);
            return CORTEX_ERROR_FILE_LOAD;
        }
    }
    
    // Calculate tensor data size
    loader->tensor_data_size = 0;
    for (uint64_t i = 0; i < loader->header.n_tensors; i++) {
        gguf_tensor_info *tensor = &loader->tensors[i];
        uint64_t nelements = 1;
        for (uint32_t j = 0; j < tensor->n_dims; j++) {
            nelements *= tensor->shape[j];
        }
        size_t type_size = ggml_type_size(tensor->type);
        size_t blck_size = ggml_blck_size(tensor->type);
        size_t tensor_size = (nelements * type_size) / blck_size;
        loader->tensor_data_size += tensor_size;
    }
    
    // Critical fix: data area needs to be aligned to alignment boundary
    // GGUF V3 uses 32-byte alignment
    const size_t alignment = 32;
    long current_pos = ftell(loader->file);
    long aligned_pos = ((current_pos + alignment - 1) / alignment) * alignment;
    
    if (fseek(loader->file, aligned_pos, SEEK_SET) != 0) {
        fclose(loader->file);
        free(loader->kv_pairs);
        free(loader->tensors);
        return CORTEX_ERROR_FILE_LOAD;
    }
    
    // Allocate tensor data memory
    loader->tensor_data = malloc(loader->tensor_data_size);
    if (!loader->tensor_data) {
        fclose(loader->file);
        free(loader->kv_pairs);
        free(loader->tensors);
        return CORTEX_ERROR_MEMORY;
    }
    
    // Read tensor data
    size_t bytes_read = fread(loader->tensor_data, 1, loader->tensor_data_size, loader->file);
    if (bytes_read != loader->tensor_data_size) {
        fclose(loader->file);
        free(loader->kv_pairs);
        free(loader->tensors);
        free(loader->tensor_data);
        return CORTEX_ERROR_FILE_LOAD;
    }
    
    return CORTEX_OK;
}

// Free GGUF loader
void free_gguf_loader(gguf_loader *loader) {
    if (loader->file) {
        fclose(loader->file);
    }
    
    if (loader->kv_pairs) {
        for (uint64_t i = 0; i < loader->header.n_kv; i++) {
            free(loader->kv_pairs[i].key);
            if (loader->kv_pairs[i].type == GGUF_TYPE_STRING) {
                free(loader->kv_pairs[i].value.string_val);
            }
        }
        free(loader->kv_pairs);
    }
    
    if (loader->tensors) {
        for (uint64_t i = 0; i < loader->header.n_tensors; i++) {
            free(loader->tensors[i].name);
        }
        free(loader->tensors);
    }
    
    if (loader->tensor_data) {
        free(loader->tensor_data);
    }
}

// Find key-value pair
gguf_kv* find_kv(gguf_loader *loader, const char *key) {
    for (uint64_t i = 0; i < loader->header.n_kv; i++) {
        if (strcmp(loader->kv_pairs[i].key, key) == 0) {
            return &loader->kv_pairs[i];
        }
    }
    return NULL;
}

// Find tensor
gguf_tensor_info* find_tensor(gguf_loader *loader, const char *name) {
    for (uint64_t i = 0; i < loader->header.n_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            return &loader->tensors[i];
        }
    }
    return NULL;
}
