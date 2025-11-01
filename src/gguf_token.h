#pragma once

#include "gguf.h"
#include "../utils/chat_template.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tokenizer structure definition 
typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
    chat_template_handle template_handle;
    bool use_chat_template;
} Tokenizer;

/**
 * Load tokenizer from GGUF file context
 * 
 * @param ctx GGUF context (must not be NULL)
 * @param t Tokenizer structure to populate
 * @param enable_thinking Whether to enable thinking mode (for template loading, currently unused)
 * @return 0 on success, non-zero on failure
 */
int load_tokenizer_from_gguf(struct gguf_context *ctx, Tokenizer *t, int enable_thinking);

#ifdef __cplusplus
}
#endif

