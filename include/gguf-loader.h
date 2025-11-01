#pragma once

#include "ggml.h"
#include "gguf.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct cortex_vocab;
struct cortex_model;
struct cortex_context;

// Type definitions
typedef int32_t cortex_token;
typedef int32_t cortex_pos;

// Model parameters
struct cortex_model_params {
    bool vocab_only;
    bool use_mmap;
    bool check_tensors;
};

// Context parameters
struct cortex_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
};

// Batch structure for token processing
struct cortex_batch {
    int32_t n_tokens;
    cortex_token *token;
    cortex_pos *pos;
    int8_t *logits;
};

// API Functions

// Backend initialization
void cortex_backend_init(void);

// Parameter initialization
struct cortex_model_params cortex_model_default_params(void);
struct cortex_context_params cortex_context_default_params(void);

// Model management
struct cortex_model * cortex_model_load_from_file(const char * path_model, struct cortex_model_params params);
void cortex_model_free(struct cortex_model * model);
const struct cortex_vocab * cortex_model_get_vocab(const struct cortex_model * model);

// Context management
struct cortex_context * cortex_new_context_with_model(struct cortex_model * model, struct cortex_context_params params);
void cortex_free(struct cortex_context * ctx);

// Tokenization
int32_t cortex_tokenize(
    const struct cortex_vocab * vocab,
    const char * text,
    int32_t text_len,
    cortex_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
);

// Inference
int32_t cortex_decode(struct cortex_context * ctx, struct cortex_batch batch);
float * cortex_get_logits(struct cortex_context * ctx);

// Vocab utilities
int32_t cortex_vocab_n_tokens(const struct cortex_vocab * vocab);

// GGML helper functions
float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);

#ifdef __cplusplus
}
#endif
