#include "cortex_llm.h"
#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Get GGUF type name
static const char* gguf_type_name(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: return "UINT8";
        case GGUF_TYPE_INT8: return "INT8";
        case GGUF_TYPE_UINT16: return "UINT16";
        case GGUF_TYPE_INT16: return "INT16";
        case GGUF_TYPE_UINT32: return "UINT32";
        case GGUF_TYPE_INT32: return "INT32";
        case GGUF_TYPE_FLOAT32: return "FLOAT32";
        case GGUF_TYPE_BOOL: return "BOOL";
        case GGUF_TYPE_STRING: return "STRING";
        case GGUF_TYPE_ARRAY: return "ARRAY";
        case GGUF_TYPE_UINT64: return "UINT64";
        case GGUF_TYPE_INT64: return "INT64";
        case GGUF_TYPE_FLOAT64: return "FLOAT64";
        default: return "UNKNOWN";
    }
}

// Get GGML type name
static const char* ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_I8: return "I8";
        case GGML_TYPE_I16: return "I16";
        case GGML_TYPE_I32: return "I32";
        default: return "UNKNOWN";
    }
}

// Detect system endianness
static const char* get_endianness() {
    uint32_t test = 0x12345678;
    uint8_t *bytes = (uint8_t*)&test;
    if (bytes[0] == 0x78) {
        return "LITTLE";
    } else {
        return "BIG";
    }
}

// Detect GGUF file endianness (by checking version field)
static const char* get_file_endianness(gguf_loader *loader) {
    // GGUF format always uses little endian, but we can verify by checking version field
    // If version field is read correctly (usually 3), then file is little endian
    // If read value is abnormally large, it might be big endian file read on little endian system
    if (loader->header.version == GGUF_VERSION) {
        return "LITTLE";
    } else if (loader->header.version > 1000) {
        // If version value is abnormally large, it might be an endianness issue
        return "BIG";
    } else {
        return "LITTLE"; // Default assumption is little endian
    }
}

// Print GGUF model information
static void print_gguf_info(gguf_loader *loader) {
    const char* file_endian = get_file_endianness(loader);
    const char* host_endian = get_endianness();
    printf("* File is %s endian, script is running on a %s endian host.\n", file_endian, host_endian);
    printf("* Dumping %lu key/value pair(s)\n", loader->header.n_kv);
    
    // Print key-value pairs
    for (uint64_t i = 0; i < loader->header.n_kv; i++) {
        gguf_kv *kv = &loader->kv_pairs[i];
        
        // For array types, show array element type and count
        if (kv->type == GGUF_TYPE_ARRAY) {
            char array_type_str[16];
            snprintf(array_type_str, sizeof(array_type_str), "[%s]", gguf_type_name(kv->value.array_val.type));
            printf("%6lu: %-8s | %8lu | %s = ", 
                   i + 1, array_type_str, kv->value.array_val.n, kv->key);
        } else {
            printf("%6lu: %-8s | %8lu | %s = ", 
                   i + 1, gguf_type_name(kv->type), 1UL, kv->key);
        }
        
        switch (kv->type) {
            case GGUF_TYPE_UINT8:
                printf("%u", kv->value.uint8_val);
                break;
            case GGUF_TYPE_INT8:
                printf("%d", kv->value.int8_val);
                break;
            case GGUF_TYPE_UINT16:
                printf("%u", kv->value.uint16_val);
                break;
            case GGUF_TYPE_INT16:
                printf("%d", kv->value.int16_val);
                break;
            case GGUF_TYPE_UINT32:
                printf("%u", kv->value.uint32_val);
                break;
            case GGUF_TYPE_INT32:
                printf("%d", kv->value.int32_val);
                break;
            case GGUF_TYPE_FLOAT32:
                printf("%.6g", kv->value.float32_val);
                break;
            case GGUF_TYPE_BOOL:
                printf("%s", kv->value.bool_val ? "True" : "False");
                break;
            case GGUF_TYPE_STRING:
                printf("'%s'", kv->value.string_val);
                break;
            case GGUF_TYPE_ARRAY:
                printf("[");
                if (kv->value.array_val.type == GGUF_TYPE_STRING) {
                    char **str_array = (char**)kv->value.array_val.data;
                    printf("'%s'", str_array[0]);
                    for (uint64_t j = 1; j < kv->value.array_val.n && j < 3; j++) {
                        printf(", '%s'", str_array[j]);
                    }
                    if (kv->value.array_val.n > 3) {
                        printf(", ...");
                    }
                } else if (kv->value.array_val.type == GGUF_TYPE_INT32) {
                    int32_t *int_array = (int32_t*)kv->value.array_val.data;
                    printf("%d", int_array[0]);
                    for (uint64_t j = 1; j < kv->value.array_val.n && j < 3; j++) {
                        printf(", %d", int_array[j]);
                    }
                    if (kv->value.array_val.n > 3) {
                        printf(", ...");
                    }
                } else {
                    printf("%lu items", kv->value.array_val.n);
                }
                printf("]");
                break;
            case GGUF_TYPE_UINT64:
                printf("%lu", kv->value.uint64_val);
                break;
            case GGUF_TYPE_INT64:
                printf("%ld", kv->value.int64_val);
                break;
            case GGUF_TYPE_FLOAT64:
                printf("%.6g", kv->value.float64_val);
                break;
            default:
                printf("UNKNOWN");
                break;
        }
        printf("\n");
    }
    
    // Print tensor information
    printf("* Dumping %lu tensor(s)\n", loader->header.n_tensors);
    for (uint64_t i = 0; i < loader->header.n_tensors; i++) {
        gguf_tensor_info *tensor = &loader->tensors[i];
        size_t tensor_size = ggml_type_size(tensor->type);
        for (uint32_t d = 0; d < tensor->n_dims; d++) {
            tensor_size *= tensor->shape[d];
        }
        
        // Ensure dimension display format matches gguf_dump.txt
        uint32_t dims[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < tensor->n_dims && d < 4; d++) {
            dims[d] = tensor->shape[d];
        }
        
        printf("%6lu: %10lu | %4u, %4u, %4u, %4u | %-6s | %s\n",
               i + 1, tensor_size,
               dims[0], dims[1], dims[2], dims[3],
               ggml_type_name(tensor->type), tensor->name);
    }
}

// Create tensor from GGUF loader
static cortex_tensor* create_tensor_from_gguf(gguf_loader *loader, gguf_tensor_info *tensor_info) {
    if (!tensor_info) {
        printf("Error: tensor_info is NULL\n");
        return NULL;
    }
    
    cortex_tensor *tensor = malloc(sizeof(cortex_tensor));
    if (!tensor) return NULL;
    
    strncpy(tensor->name, tensor_info->name, sizeof(tensor->name) - 1);
    tensor->name[sizeof(tensor->name) - 1] = '\0';
    
    tensor->type = tensor_info->type;
    
    // Set dimensions
    for (int i = 0; i < 4; i++) {
        if (i < (int)tensor_info->n_dims) {
            tensor->ne[i] = tensor_info->shape[i];
        } else {
            tensor->ne[i] = 1;
        }
    }
    
    // Calculate strides - special handling needed for quantized types
    size_t type_size = ggml_type_size(tensor_info->type);
    size_t blck_size = ggml_blck_size(tensor_info->type);
    
    tensor->nb[0] = type_size;
    // For quantized types, nb[1] = nb[0] * (ne[0] / blck_size)
    tensor->nb[1] = tensor->nb[0] * (tensor->ne[0] / blck_size);
    for (int i = 2; i < 4; i++) {
        tensor->nb[i] = tensor->nb[i-1] * tensor->ne[i-1];
    }
    
    // Set data pointer
    tensor->data = (char*)loader->tensor_data + tensor_info->offset;
    
    return tensor;
}

// Load model hyperparameters
static cortex_error load_hparams(gguf_loader *loader, cortex_hparams *hparams) {
    gguf_kv *kv;
    
    // Set default values (will be overwritten by values from GGUF)
    hparams->n_vocab = 32000;  // Default value, should be read from tokenizer.ggml.tokens
    hparams->n_ctx = 2048;
    hparams->n_embd = 4096;
    hparams->n_head = 32;
    hparams->n_head_kv = 32;
    hparams->n_layer = 32;
    hparams->n_ff = 11008;
    hparams->n_rot = 128;
    hparams->rope_freq_base = 10000.0f;
    hparams->rope_freq_scale = 1.0f;
    hparams->use_parallel_residual = false;
    
    // Read parameters from GGUF file - support Qwen3 model
    if ((kv = find_kv(loader, "qwen3.block_count")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_layer = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.block_count")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_layer = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.context_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_ctx = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.context_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_ctx = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.embedding_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_embd = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.embedding_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_embd = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.attention.head_count")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_head = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.attention.head_count")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_head = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.attention.head_count_kv")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_head_kv = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.attention.head_count_kv")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_head_kv = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.feed_forward_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_ff = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.feed_forward_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_ff = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.attention.key_length")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_rot = kv->value.uint32_val;
    } else if ((kv = find_kv(loader, "llama.rope.dimension_count")) && kv->type == GGUF_TYPE_UINT32) {
        hparams->n_rot = kv->value.uint32_val;
    }
    
    if ((kv = find_kv(loader, "qwen3.rope.freq_base")) && kv->type == GGUF_TYPE_FLOAT32) {
        hparams->rope_freq_base = kv->value.float32_val;
    } else if ((kv = find_kv(loader, "llama.rope.freq_base")) && kv->type == GGUF_TYPE_FLOAT32) {
        hparams->rope_freq_base = kv->value.float32_val;
    }
    
    if ((kv = find_kv(loader, "llama.rope.scale_linear")) && kv->type == GGUF_TYPE_FLOAT32) {
        hparams->rope_freq_scale = kv->value.float32_val;
    }
    
    // Check if it's Qwen architecture
    if ((kv = find_kv(loader, "general.architecture")) && kv->type == GGUF_TYPE_STRING) {
        if (strcmp(kv->value.string_val, "qwen2") == 0 || strcmp(kv->value.string_val, "qwen2.5") == 0) {
            hparams->use_parallel_residual = true;
        }
    }
    
    // Read vocabulary size from tokenizer.ggml.tokens array size
    if ((kv = find_kv(loader, "tokenizer.ggml.tokens")) && kv->type == GGUF_TYPE_ARRAY) {
        hparams->n_vocab = kv->value.array_val.n;
    }
    
    
    return CORTEX_OK;
}

// Load vocabulary
static cortex_error load_vocab(gguf_loader *loader, cortex_vocab *vocab, uint32_t n_vocab) {
    // Allocate vocabulary memory
    vocab->tokens = malloc(n_vocab * sizeof(char*));
    vocab->scores = malloc(n_vocab * sizeof(float));
    if (!vocab->tokens || !vocab->scores) {
        return CORTEX_ERROR_MEMORY;
    }
    
    vocab->size = n_vocab;
    
    // Set special tokens
    vocab->bos_token = 1;
    vocab->eos_token = 2;
    vocab->unk_token = 0;
    
    // Read vocabulary from GGUF file
    gguf_kv *kv_tokens = find_kv(loader, "tokenizer.ggml.tokens");
    gguf_kv *kv_scores = find_kv(loader, "tokenizer.ggml.scores");
    
    if (kv_tokens && kv_tokens->type == GGUF_TYPE_ARRAY) {
        // Read token string array
        for (uint32_t i = 0; i < n_vocab && i < kv_tokens->value.array_val.n; i++) {
            char **token_array = (char**)kv_tokens->value.array_val.data;
            size_t token_len = strlen(token_array[i]);
            vocab->tokens[i] = malloc(token_len + 1);
            if (vocab->tokens[i]) {
                strcpy(vocab->tokens[i], token_array[i]);
            }
        }
    }
    
    if (kv_scores && kv_scores->type == GGUF_TYPE_ARRAY) {
        // Read token score array
        float *score_array = (float*)kv_scores->value.array_val.data;
        for (uint32_t i = 0; i < n_vocab && i < kv_scores->value.array_val.n; i++) {
            vocab->scores[i] = score_array[i];
        }
    }
    
    return CORTEX_OK;
}

// Load model layer
static cortex_error load_layer(gguf_loader *loader, cortex_layer *layer, int layer_idx) {
    char name_buf[256];
    
    // Load attention layer weights
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", layer_idx);
    layer->attn_norm = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    // Qwen3 doesn't have attn_norm.bias tensor
    layer->attn_norm_b = NULL;
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.weight", layer_idx);
    layer->wq = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k.weight", layer_idx);
    layer->wk = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_v.weight", layer_idx);
    layer->wv = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output.weight", layer_idx);
    layer->wo = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    // Qwen3-specific Q and K normalization layers (shape=[128], per-head)
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q_norm.weight", layer_idx);
    gguf_tensor_info* q_norm_info = find_tensor(loader, name_buf);
    if (q_norm_info) {
        layer->attn_q_norm = create_tensor_from_gguf(loader, q_norm_info);
    } else {
        layer->attn_q_norm = NULL;
    }
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k_norm.weight", layer_idx);
    gguf_tensor_info* k_norm_info = find_tensor(loader, name_buf);
    if (k_norm_info) {
        layer->attn_k_norm = create_tensor_from_gguf(loader, k_norm_info);
    } else {
        layer->attn_k_norm = NULL;
    }
    
    // Qwen3-specific attention output normalization layer (shape=[n_embd_q])
    // Try multiple possible names
    const char* sub_norm_names[] = {
        "blk.%d.attn_sub_norm.weight",
        "blk.%d.post_attention_layernorm.weight", 
        "blk.%d.attn_out_norm.weight",
        NULL
    };
    
    layer->attn_sub_norm = NULL;
    for (int i = 0; sub_norm_names[i] != NULL; i++) {
        snprintf(name_buf, sizeof(name_buf), sub_norm_names[i], layer_idx);
        gguf_tensor_info* sub_norm_info = find_tensor(loader, name_buf);
        if (sub_norm_info) {
            layer->attn_sub_norm = create_tensor_from_gguf(loader, sub_norm_info);
            break;
        }
    }
    
    // Load feed-forward network weights
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", layer_idx);
    layer->ffn_norm = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    
    // Qwen3 doesn't have ffn_norm.bias tensor
    layer->ffn_norm_b = NULL;
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up.weight", layer_idx);
    layer->ffn_up = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", layer_idx);
    layer->ffn_gate = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", layer_idx);
    layer->ffn_down = create_tensor_from_gguf(loader, find_tensor(loader, name_buf));
    
    return CORTEX_OK;
}

// Main model loading function
cortex_error cortex_model_load(const char *model_path, cortex_model **model) {
    *model = malloc(sizeof(cortex_model));
    if (!*model) {
        return CORTEX_ERROR_MEMORY;
    }
    
    // Initialize model
    memset(*model, 0, sizeof(cortex_model));
    
    // Load GGUF file
    gguf_loader loader;
    cortex_error err = load_gguf_file(model_path, &loader);
    if (err != CORTEX_OK) {
        free(*model);
        return err;
    }
    
    // Print GGUF model information
    print_gguf_info(&loader);
    
    // Load hyperparameters
    err = load_hparams(&loader, &(*model)->hparams);
    if (err != CORTEX_OK) {
        free_gguf_loader(&loader);
        free(*model);
        return err;
    }
    
    // Load vocabulary
    err = load_vocab(&loader, &(*model)->vocab, (*model)->hparams.n_vocab);
    if (err != CORTEX_OK) {
        free_gguf_loader(&loader);
        free(*model);
        return err;
    }
    
    // Allocate layer array
    (*model)->layers = malloc((*model)->hparams.n_layer * sizeof(cortex_layer));
    if (!(*model)->layers) {
        free_gguf_loader(&loader);
        free(*model);
        return CORTEX_ERROR_MEMORY;
    }
    
    // Load model layers (only load existing layers)
    for (uint32_t i = 0; i < (*model)->hparams.n_layer; i++) {
        err = load_layer(&loader, &(*model)->layers[i], i);
        if (err != CORTEX_OK) {
            // Continue loading other layers, don't exit
            continue;
        }
    }
    
    // Load embedding and output layers 
    (*model)->tok_embeddings = create_tensor_from_gguf(&loader, find_tensor(&loader, "token_embd.weight"));
    (*model)->output_norm = create_tensor_from_gguf(&loader, find_tensor(&loader, "output_norm.weight"));
    (*model)->output = create_tensor_from_gguf(&loader, find_tensor(&loader, "output.weight"));
    
    
    // Qwen3 doesn't have output_norm.bias tensor
    (*model)->output_norm_b = NULL;
    
    // Save model data
    (*model)->model_data = loader.tensor_data;
    (*model)->model_size = loader.tensor_data_size;
    
    // Clean up loader (but keep tensor data)
    loader.tensor_data = NULL; // Prevent from being freed
    free_gguf_loader(&loader);
    
    return CORTEX_OK;
}

// Free model
void cortex_model_free(cortex_model *model) {
    if (!model) return;
    
    // Free vocabulary
    if (model->vocab.tokens) {
        for (uint32_t i = 0; i < model->vocab.size; i++) {
            free(model->vocab.tokens[i]);
        }
        free(model->vocab.tokens);
    }
    free(model->vocab.scores);
    
    // Free layers
    if (model->layers) {
        for (uint32_t i = 0; i < model->hparams.n_layer; i++) {
            cortex_layer *layer = &model->layers[i];
            free(layer->attn_norm);
            free(layer->attn_norm_b);
            free(layer->wq);
            free(layer->wk);
            free(layer->wv);
            free(layer->wo);
            free(layer->bq);
            free(layer->bk);
            free(layer->bv);
            free(layer->bo);
            free(layer->ffn_norm);
            free(layer->ffn_norm_b);
            free(layer->ffn_up);
            free(layer->ffn_gate);
            free(layer->ffn_down);
            free(layer->ffn_up_b);
            free(layer->ffn_gate_b);
            free(layer->ffn_down_b);
        }
        free(model->layers);
    }
    
    // Free embedding and output layers
    free(model->tok_embeddings);
    free(model->output_norm);
    free(model->output_norm_b);
    free(model->output);
    
    // Free model data
    free(model->model_data);
    
    free(model);
}
