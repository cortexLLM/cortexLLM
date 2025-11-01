#include "gguf-loader.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>

// Internal implementation structures

struct cortex_vocab_impl {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int> token_types;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int unk_token_id = -1;
    int pad_token_id = -1;
    
    cortex_vocab_impl() = default;
    ~cortex_vocab_impl() = default;
};

struct cortex_model_impl {
    struct gguf_context * gguf_ctx = nullptr;
    struct ggml_context * ggml_ctx = nullptr;
    std::unique_ptr<cortex_vocab_impl> vocab;
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    bool vocab_only = false;
    
    cortex_model_impl() = default;
    ~cortex_model_impl() {
        if (ggml_ctx) {
            ggml_free(ggml_ctx);
        }
        if (gguf_ctx) {
            gguf_free(gguf_ctx);
        }
    }
};

struct cortex_context_impl {
    cortex_model_impl * model = nullptr;
    struct cortex_context_params params;
    struct ggml_context * compute_ctx = nullptr;
    float * logits = nullptr;
    int32_t logits_size = 0;
    
    cortex_context_impl() = default;
    ~cortex_context_impl() {
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
        if (logits) {
            free(logits);
        }
    }
};

// Global backend initialization flag
static bool g_backend_initialized = false;

// API Implementation

void cortex_backend_init(void) {
    if (!g_backend_initialized) {
        // Initialize GGML backend if needed
        g_backend_initialized = true;
    }
}

struct cortex_model_params cortex_model_default_params(void) {
    struct cortex_model_params params = {};
    params.vocab_only = false;
    params.use_mmap = true;
    params.check_tensors = false;
    return params;
}

struct cortex_context_params cortex_context_default_params(void) {
    struct cortex_context_params params = {};
    params.n_ctx = 512;
    params.n_batch = 512;
    return params;
}

// Helper function to load vocab from GGUF
static bool load_vocab_from_gguf(struct gguf_context * ctx, cortex_vocab_impl * vocab) {
    // Get tokenizer tokens
    const char * tokens_key = "tokenizer.ggml.tokens";
    const char * scores_key = "tokenizer.ggml.scores";
    const char * token_types_key = "tokenizer.ggml.token_type";
    
    // Find tokens key
    int64_t tokens_key_id = gguf_find_key(ctx, tokens_key);
    if (tokens_key_id == -1) {
        printf("Warning: No tokens found in GGUF file\n");
        return false;
    }
    
    // Check if it's an array of strings
    if (gguf_get_kv_type(ctx, tokens_key_id) != GGUF_TYPE_ARRAY || 
        gguf_get_arr_type(ctx, tokens_key_id) != GGUF_TYPE_STRING) {
        printf("Warning: Invalid token type in GGUF file\n");
        return false;
    }
    
    // Load token strings
    size_t n_tokens = gguf_get_arr_n(ctx, tokens_key_id);
    vocab->tokens.resize(n_tokens);
    
    // For string arrays, we need to access each string individually
    // GGUF stores string arrays differently than other arrays
    for (size_t i = 0; i < n_tokens; i++) {
        // This is a simplified approach - in practice, we'd need to parse the string array properly
        // For now, create dummy tokens
        vocab->tokens[i] = "token_" + std::to_string(i);
    }
    
    // Load scores if available
    int64_t scores_key_id = gguf_find_key(ctx, scores_key);
    if (scores_key_id != -1 && 
        gguf_get_kv_type(ctx, scores_key_id) == GGUF_TYPE_ARRAY && 
        gguf_get_arr_type(ctx, scores_key_id) == GGUF_TYPE_FLOAT32) {
        vocab->scores.resize(n_tokens);
        const float * scores_data = (const float*)gguf_get_arr_data(ctx, scores_key_id);
        memcpy(vocab->scores.data(), scores_data, n_tokens * sizeof(float));
    } else {
        vocab->scores.resize(n_tokens, 0.0f);
    }
    
    // Load token types if available
    int64_t types_key_id = gguf_find_key(ctx, token_types_key);
    if (types_key_id != -1 && 
        gguf_get_kv_type(ctx, types_key_id) == GGUF_TYPE_ARRAY && 
        gguf_get_arr_type(ctx, types_key_id) == GGUF_TYPE_INT32) {
        vocab->token_types.resize(n_tokens);
        const int32_t * types_data = (const int32_t*)gguf_get_arr_data(ctx, types_key_id);
        memcpy(vocab->token_types.data(), types_data, n_tokens * sizeof(int32_t));
    } else {
        vocab->token_types.resize(n_tokens, 1); // Default to normal tokens
    }
    
    // Load special token IDs (with safe type checking)
    int64_t bos_key_id = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_key_id != -1 && gguf_get_kv_type(ctx, bos_key_id) == GGUF_TYPE_INT32) {
        vocab->bos_token_id = gguf_get_val_i32(ctx, bos_key_id);
    }
    
    int64_t eos_key_id = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_key_id != -1 && gguf_get_kv_type(ctx, eos_key_id) == GGUF_TYPE_INT32) {
        vocab->eos_token_id = gguf_get_val_i32(ctx, eos_key_id);
    }
    
    int64_t unk_key_id = gguf_find_key(ctx, "tokenizer.ggml.unknown_token_id");
    if (unk_key_id != -1 && gguf_get_kv_type(ctx, unk_key_id) == GGUF_TYPE_INT32) {
        vocab->unk_token_id = gguf_get_val_i32(ctx, unk_key_id);
    }
    
    int64_t pad_key_id = gguf_find_key(ctx, "tokenizer.ggml.padding_token_id");
    if (pad_key_id != -1 && gguf_get_kv_type(ctx, pad_key_id) == GGUF_TYPE_INT32) {
        vocab->pad_token_id = gguf_get_val_i32(ctx, pad_key_id);
    }
    
    return true;
}

struct cortex_model * cortex_model_load_from_file(const char * path_model, struct cortex_model_params params) {
    cortex_backend_init();
    
    auto * model = new cortex_model_impl();
    model->vocab_only = params.vocab_only;
    
    // Load GGUF file
    struct gguf_init_params init_params = {
        .no_alloc = false,
        .ctx = nullptr
    };
    model->gguf_ctx = gguf_init_from_file(path_model, init_params);
    if (!model->gguf_ctx) {
        printf("Failed to load GGUF file: %s\n", path_model);
        delete model;
        return nullptr;
    }
    
    // Initialize vocab
    model->vocab = std::make_unique<cortex_vocab_impl>();
    if (!load_vocab_from_gguf(model->gguf_ctx, model->vocab.get())) {
        printf("Failed to load vocab from GGUF file\n");
        delete model;
        return nullptr;
    }
    
    // If not vocab_only, load tensors
    if (!params.vocab_only) {
        // Create GGML context for tensors
        size_t ctx_size = gguf_get_meta_size(model->gguf_ctx);
        model->ggml_ctx = ggml_init({.mem_size = ctx_size * 2, .mem_buffer = nullptr});
        
        if (!model->ggml_ctx) {
            printf("Failed to create GGML context\n");
            delete model;
            return nullptr;
        }
        
        // Load tensors (simplified - just store references)
        int n_tensors = gguf_get_n_tensors(model->gguf_ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(model->gguf_ctx, i);
            struct ggml_tensor * tensor = ggml_get_tensor(model->ggml_ctx, name);
            if (tensor) {
                model->tensors[name] = tensor;
            }
        }
    }
    
    return reinterpret_cast<struct cortex_model*>(model);
}

void cortex_model_free(struct cortex_model * model) {
    if (model) {
        delete reinterpret_cast<cortex_model_impl*>(model);
    }
}

const struct cortex_vocab * cortex_model_get_vocab(const struct cortex_model * model) {
    if (!model) return nullptr;
    auto * impl = reinterpret_cast<const cortex_model_impl*>(model);
    return reinterpret_cast<const struct cortex_vocab*>(impl->vocab.get());
}

struct cortex_context * cortex_new_context_with_model(struct cortex_model * model, struct cortex_context_params params) {
    if (!model) return nullptr;
    
    auto * ctx = new cortex_context_impl();
    ctx->model = reinterpret_cast<cortex_model_impl*>(model);
    ctx->params = params;
    
    // Create compute context
    size_t compute_size = 1024 * 1024; // 1MB for compute
    ctx->compute_ctx = ggml_init({.mem_size = compute_size, .mem_buffer = nullptr});
    
    if (!ctx->compute_ctx) {
        printf("Failed to create compute context\n");
        delete ctx;
        return nullptr;
    }
    
    // Allocate logits buffer
    auto * vocab_impl = reinterpret_cast<cortex_model_impl*>(model)->vocab.get();
    ctx->logits_size = vocab_impl->tokens.size();
    ctx->logits = (float*)malloc(ctx->logits_size * sizeof(float));
    
    if (!ctx->logits) {
        printf("Failed to allocate logits buffer\n");
        delete ctx;
        return nullptr;
    }
    
    return reinterpret_cast<struct cortex_context*>(ctx);
}

void cortex_free(struct cortex_context * ctx) {
    if (ctx) {
        delete reinterpret_cast<cortex_context_impl*>(ctx);
    }
}

int32_t cortex_tokenize(
    const struct cortex_vocab * vocab,
    const char * text,
    int32_t text_len,
    cortex_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
) {
    if (!vocab || !text || !tokens) return -1;
    
    auto * vocab_impl = reinterpret_cast<const cortex_vocab_impl*>(vocab);
    
    // Simple tokenization: split by spaces and match against vocab
    std::string input_text(text, text_len);
    std::vector<std::string> words;
    
    // Split by spaces
    size_t start = 0;
    size_t end = input_text.find(' ');
    while (end != std::string::npos) {
        words.push_back(input_text.substr(start, end - start));
        start = end + 1;
        end = input_text.find(' ', start);
    }
    words.push_back(input_text.substr(start));
    
    int32_t token_count = 0;
    
    // Add BOS token if requested
    if (add_special && vocab_impl->bos_token_id >= 0 && token_count < n_tokens_max) {
        tokens[token_count++] = vocab_impl->bos_token_id;
    }
    
    // Tokenize each word
    for (const auto & word : words) {
        if (token_count >= n_tokens_max) break;
        
        // Try to find exact match in vocab
        bool found = false;
        for (size_t i = 0; i < vocab_impl->tokens.size(); i++) {
            if (vocab_impl->tokens[i] == word) {
                tokens[token_count++] = i;
                found = true;
                break;
            }
        }
        
        // If not found, use UNK token
        if (!found && vocab_impl->unk_token_id >= 0) {
            tokens[token_count++] = vocab_impl->unk_token_id;
        }
    }
    
    // Add EOS token if requested
    if (add_special && vocab_impl->eos_token_id >= 0 && token_count < n_tokens_max) {
        tokens[token_count++] = vocab_impl->eos_token_id;
    }
    
    return token_count;
}

int32_t cortex_decode(struct cortex_context * ctx, struct cortex_batch batch) {
    if (!ctx || !batch.token || batch.n_tokens <= 0) return -1;
    
    auto * ctx_impl = reinterpret_cast<cortex_context_impl*>(ctx);
    
    // Simplified decode: just generate dummy logits for now
    // In a real implementation, this would build and execute the transformer forward pass
    
    for (int32_t i = 0; i < ctx_impl->logits_size; i++) {
        ctx_impl->logits[i] = 0.0f;
    }
    
    // Set some dummy probabilities
    if (ctx_impl->logits_size > 0) {
        ctx_impl->logits[0] = 1.0f; // Bias towards first token
    }
    
    return 0; // Success
}

float * cortex_get_logits(struct cortex_context * ctx) {
    if (!ctx) return nullptr;
    
    auto * ctx_impl = reinterpret_cast<cortex_context_impl*>(ctx);
    return ctx_impl->logits;
}

int32_t cortex_vocab_n_tokens(const struct cortex_vocab * vocab) {
    if (!vocab) return 0;
    
    auto * vocab_impl = reinterpret_cast<const cortex_vocab_impl*>(vocab);
    return vocab_impl->tokens.size();
}

// GGML helper functions
float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!tensor) return 0.0f;
    
    // Handle different tensor types
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return ((float*)tensor->data)[i];
        case GGML_TYPE_F16:
            {
                uint16_t * data = (uint16_t*)tensor->data;
                uint16_t h = data[i];
                // Simple FP16 to FP32 conversion
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h & 0x7C00) >> 10;
                uint32_t frac = h & 0x03FF;
                
                if (exp == 0) {
                    if (frac == 0) {
                        return sign ? -0.0f : 0.0f;
                    } else {
                        // Denormalized
                        float val = (float)frac / 1024.0f;
                        return sign ? -val : val;
                    }
                } else if (exp == 31) {
                    return sign ? -HUGE_VALF : HUGE_VALF;
                } else {
                    // Normalized
                    float val = 1.0f + (float)frac / 1024.0f;
                    val *= powf(2.0f, (int)exp - 15);
                    return sign ? -val : val;
                }
            }
        case GGML_TYPE_Q8_0:
            {
                // Q8_0 format: scale + 32 quantized values
                const uint8_t * data = (const uint8_t*)tensor->data;
                int block_idx = i / 32;
                int elem_idx = i % 32;
                
                float scale = *(float*)(data + block_idx * 33);
                int8_t qval = data[block_idx * 33 + 4 + elem_idx];
                
                return scale * qval;
            }
        default:
            printf("Warning: Unsupported tensor type for ggml_get_f32_1d: %d\n", tensor->type);
            return 0.0f;
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    if (!tensor) return 0.0f;
    
    // Calculate linear index from multi-dimensional indices
    int64_t idx = i0;
    int n_dims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1) n_dims = i + 1;
    }
    
    if (n_dims > 1) idx += i1 * tensor->ne[0];
    if (n_dims > 2) idx += i2 * tensor->ne[0] * tensor->ne[1];
    if (n_dims > 3) idx += i3 * tensor->ne[0] * tensor->ne[1] * tensor->ne[2];
    
    return ggml_get_f32_1d(tensor, idx);
}
