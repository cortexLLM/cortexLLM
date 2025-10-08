#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Basic type definitions
typedef int32_t cortex_token;
typedef int32_t cortex_pos;

// Error codes
typedef enum {
    CORTEX_OK = 0,
    CORTEX_ERROR_FILE_LOAD = -1,
    CORTEX_ERROR_INVALID_MODEL = -2,
    CORTEX_ERROR_MEMORY = -3,
    CORTEX_ERROR_INVALID_PARAM = -4,
    CORTEX_ERROR_TOKENIZATION = -5,
    CORTEX_ERROR_INFERENCE = -6,
} cortex_error;

// Model hyperparameters structure
typedef struct {
    uint32_t n_vocab;      // vocabulary size
    uint32_t n_ctx;        // context length
    uint32_t n_embd;       // embedding dimension
    uint32_t n_head;       // number of attention heads
    uint32_t n_head_kv;    // number of KV heads
    uint32_t n_layer;      // number of layers
    uint32_t n_ff;         // feed-forward network dimension
    uint32_t n_rot;        // rotary position encoding dimension
    float rope_freq_base;  // RoPE frequency base
    float rope_freq_scale; // RoPE frequency scale
    bool use_parallel_residual; // whether to use parallel residual
} cortex_hparams;

// Vocabulary structure
typedef struct {
    char **tokens;         // vocabulary string array
    float *scores;         // vocabulary scores
    uint32_t size;         // vocabulary size
    cortex_token bos_token; // BOS token
    cortex_token eos_token; // EOS token
    cortex_token unk_token; // UNK token
} cortex_vocab;

// Tensor structure
typedef struct {
    void *data;            // data pointer
    int64_t ne[4];         // dimension sizes
    size_t nb[4];          // strides
    int32_t type;          // data type
    char name[64];         // tensor name
} cortex_tensor;

// Model layer structure
typedef struct {
    // attention layer
    cortex_tensor *attn_norm;      // attention layer normalization
    cortex_tensor *attn_norm_b;    // attention layer normalization bias
    cortex_tensor *wq;             // query weights
    cortex_tensor *wk;             // key weights
    cortex_tensor *wv;             // value weights
    cortex_tensor *wo;             // output weights
    cortex_tensor *bq;             // query bias
    cortex_tensor *bk;             // key bias
    cortex_tensor *bv;             // value bias
    cortex_tensor *bo;             // output bias
    
    // Qwen3-specific Q and K normalization layers
    cortex_tensor *attn_q_norm;    // Q normalization
    cortex_tensor *attn_k_norm;    // K normalization
    
    // Qwen3-specific attention output normalization layer (before wo projection)
    cortex_tensor *attn_sub_norm;  // attention output normalization
    
    // feed-forward network layer
    cortex_tensor *ffn_norm;       // FFN normalization
    cortex_tensor *ffn_norm_b;     // FFN normalization bias
    cortex_tensor *ffn_up;         // FFN up projection
    cortex_tensor *ffn_gate;       // FFN gate
    cortex_tensor *ffn_down;       // FFN down projection
    cortex_tensor *ffn_up_b;       // FFN up projection bias
    cortex_tensor *ffn_gate_b;     // FFN gate bias
    cortex_tensor *ffn_down_b;     // FFN down projection bias
} cortex_layer;

// Model structure
typedef struct {
    cortex_hparams hparams;        // model hyperparameters
    cortex_vocab vocab;            // vocabulary
    
    // embedding layer
    cortex_tensor *tok_embeddings; // token embeddings
    cortex_tensor *output_norm;    // output normalization
    cortex_tensor *output_norm_b;  // output normalization bias
    cortex_tensor *output;         // output layer
    
    // layer array
    cortex_layer *layers;          // model layer array
    
    // memory management
    void *model_data;              // model data memory
    size_t model_size;             // model data size
} cortex_model;

// Context structure
typedef struct {
    cortex_model *model;           // model pointer
    
    // computation buffers
    float *embd;                   // embedding buffer
    float *attn_buf;               // attention buffer
    float *ffn_buf;                // FFN buffer
    float *logits;                 // output logits
    
    // KV cache
    float *k_cache;                // K cache
    float *v_cache;                // V cache
    cortex_pos n_ctx_used;         // used context length
    
    // computation parameters
    int n_threads;                 // number of threads
} cortex_context;

// API function declarations

// Model loading
cortex_error cortex_model_load(const char *model_path, cortex_model **model);

// Model freeing
void cortex_model_free(cortex_model *model);

// Context creation
cortex_error cortex_context_new(cortex_model *model, cortex_context **ctx);

// Context freeing
void cortex_context_free(cortex_context *ctx);

// Tokenization
cortex_error cortex_tokenize(cortex_model *model, const char *text, cortex_token *tokens, int max_tokens, int *n_tokens);

// Inference
cortex_error cortex_eval(cortex_context *ctx, const cortex_token *tokens, int n_tokens, cortex_pos n_past);

// Get logits
float* cortex_get_logits(cortex_context *ctx);

// Sampling
cortex_token cortex_sample(cortex_context *ctx, float temperature, float top_p, int top_k);

// Get token text
const char* cortex_token_to_str(cortex_model *model, cortex_token token);

// Get error message
const char* cortex_error_to_str(cortex_error error);

#ifdef __cplusplus
}
#endif
