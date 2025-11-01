#include "gguf_token.h"
#include "../utils/chat_template.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <string>

// Helper function to compute pseudo-scores from merges (C++ implementation)
static void compute_scores_from_merges(
    struct gguf_context *ctx,
    int merges_idx,
    char **vocab,
    float *merge_scores,
    uint32_t vocab_size)
{
    // Initialize all scores to -1e6 (initial vocab tokens)
    for (uint32_t i = 0; i < vocab_size; i++) {
        merge_scores[i] = -1e6f;
    }
    
    // Build merge rank map
    std::unordered_map<std::string, int> merge_rank;
    const uint32_t n_merges = gguf_get_arr_n(ctx, merges_idx);
    
    for (uint32_t i = 0; i < n_merges; i++) {
        const char *merge_str = gguf_get_arr_str(ctx, merges_idx, i);
        if (!merge_str) continue;
        
        // Parse merge rule: "first second" -> token "firstsecond"
        std::string merge_rule(merge_str);
        size_t space_pos = merge_rule.find(' ', 1);
        
        if (space_pos != std::string::npos) {
            std::string first = merge_rule.substr(0, space_pos);
            std::string second = merge_rule.substr(space_pos + 1);
            std::string merged = first + second;
            
            // Store rank for this merged token
            merge_rank[merged] = i;
        }
    }
    
    // Compute scores for each token
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (vocab[i]) {
            std::string token(vocab[i]);
            auto it = merge_rank.find(token);
            if (it != merge_rank.end()) {
                // Token was created by a merge, use -log(rank + 1)
                int rank = it->second;
                merge_scores[i] = -logf((float)(rank + 1));
            }
            // else: keep -1e6 (initial vocab token)
        }
    }
}

extern "C" {

int load_tokenizer_from_gguf(struct gguf_context *ctx, Tokenizer *t, int enable_thinking) {
    if (!ctx || !t) {
        fprintf(stderr, "load_tokenizer_from_gguf: NULL context or tokenizer\n");
        return 1;
    }
    
    // Find token list (required)
    const int token_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (token_idx < 0) {
        fprintf(stderr, "Missing tokenizer.ggml.tokens in GGUF file\n");
        return 1;
    }
    
    // Verify it's an array
    if (gguf_get_kv_type(ctx, token_idx) != GGUF_TYPE_ARRAY) {
        fprintf(stderr, "tokenizer.ggml.tokens is not an array\n");
        return 1;
    }
    
    // Verify array type is string
    if (gguf_get_arr_type(ctx, token_idx) != GGUF_TYPE_STRING) {
        fprintf(stderr, "tokenizer.ggml.tokens array is not string type\n");
        return 1;
    }
    
    // Get vocabulary size
    const uint32_t n_vocab = gguf_get_arr_n(ctx, token_idx);
    if (n_vocab == 0) {
        fprintf(stderr, "Tokenizer vocabulary is empty\n");
        return 1;
    }
    
    t->vocab_size = (int)n_vocab;
    
    // Allocate memory for vocab and merge_scores
    t->vocab = (char**)malloc(n_vocab * sizeof(char*));
    t->merge_scores = (float*)malloc(n_vocab * sizeof(float));
    
    if (!t->vocab || !t->merge_scores) {
        fprintf(stderr, "Failed to allocate memory for tokenizer\n");
        if (t->vocab) free(t->vocab);
        if (t->merge_scores) free(t->merge_scores);
        return 1;
    }
    
    // Initialize max_token_length
    t->max_token_length = 0;
    
    // Load tokens
    for (uint32_t i = 0; i < n_vocab; i++) {
        const char *word = gguf_get_arr_str(ctx, token_idx, i);
        if (!word) {
            fprintf(stderr, "Failed to get token %u from GGUF\n", i);
            // Clean up on error
            for (uint32_t j = 0; j < i; j++) {
                free(t->vocab[j]);
            }
            free(t->vocab);
            free(t->merge_scores);
            return 1;
        }
        
        size_t len = strlen(word);
        t->vocab[i] = (char*)malloc(len + 1);
        if (!t->vocab[i]) {
            fprintf(stderr, "Failed to allocate memory for token %u\n", i);
            // Clean up on error
            for (uint32_t j = 0; j < i; j++) {
                free(t->vocab[j]);
            }
            free(t->vocab);
            free(t->merge_scores);
            return 1;
        }
        strcpy(t->vocab[i], word);
        
        // Update max_token_length
        if (len > t->max_token_length) {
            t->max_token_length = (unsigned int)len;
        }
    }
    
    // Load merge scores
    const int score_idx = gguf_find_key(ctx, "tokenizer.ggml.scores");
    if (score_idx >= 0) {
        // Verify it's a float32 array
        if (gguf_get_kv_type(ctx, score_idx) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(ctx, score_idx) == GGUF_TYPE_FLOAT32) {
            uint32_t n_scores = gguf_get_arr_n(ctx, score_idx);
            if (n_scores == n_vocab) {
                // Direct copy scores
                const float *scores_data = (const float*)gguf_get_arr_data(ctx, score_idx);
                memcpy(t->merge_scores, scores_data, n_vocab * sizeof(float));
            } else {
                fprintf(stderr, "Warning: scores array size (%u) != vocab size (%u), initializing to 0\n", n_scores, n_vocab);
                memset(t->merge_scores, 0, n_vocab * sizeof(float));
            }
        } else {
            fprintf(stderr, "Warning: tokenizer.ggml.scores has wrong type, initializing to 0\n");
            memset(t->merge_scores, 0, n_vocab * sizeof(float));
        }
    } else {
        // No scores, try to compute from merges
        const int merges_idx = gguf_find_key(ctx, "tokenizer.ggml.merges");
        if (merges_idx >= 0) {
            // Verify it's a string array
            if (gguf_get_kv_type(ctx, merges_idx) == GGUF_TYPE_ARRAY &&
                gguf_get_arr_type(ctx, merges_idx) == GGUF_TYPE_STRING) {
                compute_scores_from_merges(ctx, merges_idx, t->vocab, t->merge_scores, n_vocab);
            } else {
                fprintf(stderr, "Warning: tokenizer.ggml.merges has wrong type, initializing scores to 0\n");
                memset(t->merge_scores, 0, n_vocab * sizeof(float));
            }
        } else {
            // No scores and no merges, initialize to 0
            fprintf(stderr, "Warning: No tokenizer.ggml.scores or tokenizer.ggml.merges found, initializing scores to 0\n");
            memset(t->merge_scores, 0, n_vocab * sizeof(float));
        }
    }
    
    // Load special token IDs
    const int bos_idx = gguf_find_key(ctx, "tokenizer.ggml.bos_token_id");
    if (bos_idx >= 0) {
        if (gguf_get_kv_type(ctx, bos_idx) == GGUF_TYPE_UINT32 || 
            gguf_get_kv_type(ctx, bos_idx) == GGUF_TYPE_INT32) {
            t->bos_token_id = gguf_get_val_u32(ctx, bos_idx);
        } else {
            t->bos_token_id = 0;
        }
    } else {
        t->bos_token_id = 0;
    }
    
    const int eos_idx = gguf_find_key(ctx, "tokenizer.ggml.eos_token_id");
    if (eos_idx >= 0) {
        if (gguf_get_kv_type(ctx, eos_idx) == GGUF_TYPE_UINT32 || 
            gguf_get_kv_type(ctx, eos_idx) == GGUF_TYPE_INT32) {
            t->eos_token_id = gguf_get_val_u32(ctx, eos_idx);
        } else {
            t->eos_token_id = 0;
        }
    } else {
        t->eos_token_id = 0;
    }
    
    // Load chat template
    const int tmpl_idx = gguf_find_key(ctx, "tokenizer.chat_template");
    
    // Get BOS/EOS token strings
    const char* bos_token_str = NULL;
    const char* eos_token_str = NULL;
    if (t->bos_token_id < (unsigned int)t->vocab_size && t->vocab[t->bos_token_id]) {
        bos_token_str = t->vocab[t->bos_token_id];
    }
    if (t->eos_token_id < (unsigned int)t->vocab_size && t->vocab[t->eos_token_id]) {
        eos_token_str = t->vocab[t->eos_token_id];
    }
    
    // Initialize template
    t->template_handle = NULL;
    t->use_chat_template = false;
    memset(t->prompt_template, 0, sizeof(t->prompt_template));
    memset(t->system_prompt_template, 0, sizeof(t->system_prompt_template));
    
    if (tmpl_idx >= 0) {
        if (gguf_get_kv_type(ctx, tmpl_idx) == GGUF_TYPE_STRING) {
            const char *tmpl = gguf_get_val_str(ctx, tmpl_idx);
            if (tmpl && strlen(tmpl) > 0) {
                // Try to load Jinja2 template
                t->template_handle = chat_template_load_from_string(tmpl, bos_token_str, eos_token_str);
                if (t->template_handle) {
                    t->use_chat_template = true;
                    printf("Loaded Jinja2 chat template from GGUF\n");
                } else {
                    // Fallback to sprintf mode
                    snprintf(t->prompt_template, sizeof(t->prompt_template), "%s", tmpl);
                    fprintf(stderr, "Warning: Failed to load Jinja2 template, falling back to sprintf mode\n");
                }
            }
        }
    }
    
    // If no template or load failed, set default
    if (!t->use_chat_template) {
        if (tmpl_idx < 0) {
            // Default Qwen-style template
            strcpy(t->prompt_template, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n");
        }
    }
    
    printf("Loaded tokenizer from GGUF: vocab_size=%d, max_token_length=%u, bos=%u, eos=%u\n",
           t->vocab_size, t->max_token_length, t->bos_token_id, t->eos_token_id);
    
    return 0;
}

} // extern "C"

