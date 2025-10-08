#include "cortex_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// GGML type definitions (copied from gguf_loader.c)
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

// Quantization function declarations
static void dequantize_q8_0(const void *src, float *dst, int n);
static void dequantize_q4_0(const void *src, float *dst, int n);

// Matrix multiplication functions
static void matmul_f32(const float *a, const float *b, float *c, int m, int n, int k);
static void matmul_q8_0_f32(const void *a, const float *b, float *c, int m, int n, int k);
static void matmul_q4_0_f32(const void *a, const float *b, float *c, int m, int n, int k);

// Activation functions
static void silu(float *x, int n);

// Normalization functions
static void rms_norm(float *x, const float *weight, const float *bias, int n, float eps);

// RoPE position encoding
static void apply_rope(float *q, float *k, int head_dim, int seq_len, int pos, float freq_base, float freq_scale);

// Attention computation
static void compute_attention(float *q, float *k, float *v, float *out, 
                             int n_head, int n_head_kv, int head_dim, int seq_len, int pos,
                             float *k_cache, float *v_cache, float freq_base, float freq_scale,
                             float *q_norm_weight, float *k_norm_weight);

// Feed-forward network
static void compute_ffn(float *x, const cortex_tensor *up, const cortex_tensor *gate, const cortex_tensor *down,
                       float *temp1, float *temp2, int n_embd, int n_ff, int layer_idx, int diag_i);


// Helper function: create float from bit representation
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

// FP16 to FP32 - completely following llama.cpp/ggml implementation
static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}


// Quantization function implementation
static void dequantize_q8_0(const void *src, float *dst, int n) {
    typedef struct {
        uint16_t d;       // FP16 delta
        int8_t qs[32];    // quantized values
    } __attribute__((packed)) block_q8_0;
    
    const block_q8_0 *blocks = (const block_q8_0*)src;
    const int nb = n / 32;  // number of blocks
    
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = blocks[i].qs[j] * d;
        }
    }
}

static void dequantize_q4_0(const void *src, float *dst, int n) {
    const uint8_t *q = (const uint8_t*)src;
    for (int i = 0; i < n; i += 32) {
        float d = *(float*)(q + i * 18 / 32);  // scale at the beginning of block
        for (int j = 0; j < 32 && i + j < n; j++) {
            int idx = j / 2;
            int shift = (j % 2) * 4;
            int8_t val = (q[i * 18 / 32 + 4 + idx] >> shift) & 0xF;
            if (val > 7) val -= 16;
            dst[i + j] = d * val;
        }
    }
}

// Matrix multiplication implementation - using double precision accumulator
static void matmul_f32(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;  // Use double precision accumulation!
            for (int l = 0; l < k; l++) {
                sum += (double)(a[i * k + l] * b[l * n + j]);
            }
            c[i * n + j] = (float)sum;
        }
    }
}

// Q8_0 matmul - correct Q8_0 × F32 matrix multiplication implementation
// Reference llama.cpp WebGPU implementation: dequantize first, then multiply with F32
static void matmul_q8_0_f32(const void *a, const float *b, float *c, int m, int n, int k) {
    // Q8_0 block structure
    typedef struct {
        uint16_t d;       // FP16 delta
        int8_t qs[32];    // quantized values
    } __attribute__((packed)) block_q8_0;
    
    const int qk = 32;  // Q8_0 block size
    const int nb = k / qk;  // number of blocks per row
    
    // Calculate row stride (bytes)
    const size_t row_size = nb * sizeof(block_q8_0);  // nb * 34
    
    for (int i = 0; i < m; i++) {
        // Use byte offset to access row i
        const char *row_data = (const char*)a + i * row_size;
        
        for (int j = 0; j < n; j++) {
            double sumd = 0.0;  // Use double precision accumulation!
            
            // Iterate through each block
            for (int ib = 0; ib < nb; ib++) {
                const block_q8_0 *block = (const block_q8_0*)(row_data + ib * sizeof(block_q8_0));
                
                // Convert delta (FP16 -> FP32)
                float d = fp16_to_fp32(block->d);
                
                // Correct Q8_0 × F32 calculation: dequantize first, then multiply with F32
                // This is consistent with llama.cpp WebGPU implementation
                for (int l = 0; l < qk; l++) {
                    int8_t q_val = block->qs[l];
                    float b_val = b[j * k + ib * qk + l];
                    float dequantized = (float)q_val * d;  // dequantize
                    sumd += (double)(dequantized * b_val);  // multiply with F32
                }
            }
            
            c[i * n + j] = (float)sumd;
        }
    }
}

// Q4_0 matmul - using double precision accumulator
static void matmul_q4_0_f32(const void *a, const float *b, float *c, int m, int n, int k) {
    float *temp = malloc(k * sizeof(float));
    if (!temp) return;
    
    for (int i = 0; i < m; i++) {
        dequantize_q4_0((char*)a + i * k * 18, temp, k);  // Q4_0 block size is 18
        for (int j = 0; j < n; j++) {
            double sum = 0.0;  // Use double precision accumulation!
            for (int l = 0; l < k; l++) {
                sum += (double)(temp[l] * b[l * n + j]);
            }
            c[i * n + j] = (float)sum;
        }
    }
    
    free(temp);
}

// Activation functions implementation
static void silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}


// RMS normalization implementation
static void rms_norm_only(float *x, int n, float eps) {
    double sum = 0.0;  // Use double precision!
    for (int i = 0; i < n; i++) {
        sum += (double)(x[i] * x[i]);  // Force conversion to double
    }
    float mean = (float)(sum / n);
    float scale = 1.0f / sqrtf(mean + eps);
    
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

// RMS normalization: complete version (normalization + apply weights)
static void rms_norm(float *x, const float *weight, const float *bias, int n, float eps) {
    // Step 1: normalization
    rms_norm_only(x, n, eps);
    
    // Step 2: apply weights and bias
    if (weight) {
    for (int i = 0; i < n; i++) {
            x[i] = x[i] * weight[i] + (bias ? bias[i] : 0.0f);
    }
    }
}
    
// Layer normalization implementation (for Q/K normalization)
// Layer normalization - using double precision accumulation, matching llama.cpp

// RoPE position encoding implementation (consistent progressive multiplication with llama.cpp)
static void apply_rope(float *q, float *k, int head_dim, int seq_len __attribute__((unused)), int pos, float freq_base, float freq_scale) {
    // Use qwen3.c RoPE implementation
    for (int j = 0; j < head_dim/2; j++) {
        float freq = powf(1e6, -(float)j / (head_dim/2));
        float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);
        
        if (q != NULL) {
            float x = q[j]; // real part
            float y = q[j + head_dim/2]; // imag part
            
            q[j] = x * cos_freq - y * sin_freq; // new real
            q[j + head_dim/2] = x * sin_freq + y * cos_freq; // new imag
        }
        
        if (k != NULL) {
            float x = k[j];
            float y = k[j + head_dim/2];
            
            k[j] = x * cos_freq - y * sin_freq;
            k[j + head_dim/2] = x * sin_freq + y * cos_freq;
        }
    }
}

// Attention computation implementation - fix Qwen3 Q/K normalization
static void compute_attention(float *q, float *k, float *v, float *out, 
                             int n_head, int n_head_kv, int head_dim, int seq_len, int pos,
                             float *k_cache, float *v_cache, float freq_base, float freq_scale,
                             float *q_norm_weight, float *k_norm_weight) {
    int n_embd_kv = n_head_kv * head_dim;
    int n_rep = n_head / n_head_kv;  // GQA repetition factor
    
    (void)seq_len;  // avoid unused warning
    
    // Compute Q, K, V
    float *Q = malloc(n_head * head_dim * sizeof(float));
    float *K = malloc(n_head_kv * head_dim * sizeof(float));  // Only n_head_kv heads
    float *V = malloc(n_head_kv * head_dim * sizeof(float));  // Only n_head_kv heads
    
    
    if (!Q || !K || !V) {
        free(Q); free(K); free(V);
        return;
    }
    
    // Copy original Q and K
    memcpy(Q, q, n_head * head_dim * sizeof(float));
    memcpy(K, k, n_head_kv * head_dim * sizeof(float));
    
        // Qwen3 key fix: Perform Q/K normalization and RoPE exactly according to llama.cpp's build_norm logic
        // 1. Apply RMS normalization + weight to Q, then apply RoPE
        if (q_norm_weight) {
            for (int h = 0; h < n_head; h++) {
                float *qh = Q + h * head_dim;
                // According to llama.cpp's build_norm logic: normalize first, then apply weights
                rms_norm_only(qh, head_dim, 1e-6f);
                for (int d = 0; d < head_dim; d++) {
                    qh[d] *= q_norm_weight[d];
                }
            }
        }
        
        // 2. Apply RoPE to Q
        for (int h = 0; h < n_head; h++) {
            float *qh = Q + h * head_dim;
            apply_rope(qh, NULL, head_dim, seq_len, pos, freq_base, freq_scale);
        }
        
        // 3. Apply RMS normalization + weight to K, then apply RoPE
        if (k_norm_weight) {
            for (int h = 0; h < n_head_kv; h++) {
                float *kh = K + h * head_dim;
                // According to llama.cpp's build_norm logic: normalize first, then apply weights
                rms_norm_only(kh, head_dim, 1e-6f);
                for (int d = 0; d < head_dim; d++) {
                    kh[d] *= k_norm_weight[d];
                }
            }
        }
        
        // 4. Apply RoPE to K
        for (int h = 0; h < n_head_kv; h++) {
            float *kh = K + h * head_dim;
            apply_rope(NULL, kh, head_dim, seq_len, pos, freq_base, freq_scale);
        }
    
    // Copy V (no RoPE needed)
    memcpy(V, v, n_head_kv * head_dim * sizeof(float));
    
    
    
    
    // Update KV cache (only store n_head_kv heads)
    if (k_cache && v_cache) {
        memcpy(k_cache + pos * n_embd_kv, K, n_embd_kv * sizeof(float));
        memcpy(v_cache + pos * n_embd_kv, V, n_embd_kv * sizeof(float));
    }
    
    // Compute attention scores (using GQA)
    float *scores = malloc(n_head * seq_len * sizeof(float));
    if (!scores) {
        free(Q); free(K); free(V);
        return;
    }
    
    for (int h = 0; h < n_head; h++) {
        float *qh = Q + h * head_dim;
        int kv_head = h / n_rep;  
        for (int t = 0; t < seq_len; t++) {
            // Causal mask: can only attend to current position and previous tokens
            if (t > pos) {
                scores[h * seq_len + t] = -INFINITY;
                continue;
            }
            
            float *kt = (k_cache ? k_cache + t * n_embd_kv : K) + kv_head * head_dim;
            double score = 0.0;  
            for (int d = 0; d < head_dim; d++) {
                score += (double)(qh[d] * kt[d]);  
            }
            // Use qwen3.c's scale factor: 1.0f/sqrtf(head_dim), where head_dim = 128
            int head_dim = 128;  // Qwen3's head_dim is fixed at 128
            scores[h * seq_len + t] = (float)(score / sqrtf(head_dim));
            
        }
    }
    
    // Softmax
    for (int h = 0; h < n_head; h++) {
        float *h_scores = scores + h * seq_len;
        float max_score = h_scores[0];
        for (int t = 1; t < seq_len; t++) {
            if (h_scores[t] > max_score) max_score = h_scores[t];
        }
        
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            h_scores[t] = expf(h_scores[t] - max_score);
            sum += h_scores[t];
        }
        
        for (int t = 0; t < seq_len; t++) {
            h_scores[t] /= sum;
        }
    }
    
    
    // Compute output (using GQA)
    for (int h = 0; h < n_head; h++) {
        float *out_h = out + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        int kv_head = h / n_rep;  // GQA: Every n_rep Q heads share one K/V head
        
        for (int t = 0; t < seq_len; t++) {
            float *vt = (v_cache ? v_cache + t * n_embd_kv : V) + kv_head * head_dim;
            float score = scores[h * seq_len + t];
            
            for (int d = 0; d < head_dim; d++) {
                out_h[d] += (float)((double)score * (double)vt[d]);  // Use double precision calculation
            }
        }
    }
    
    
    free(Q); free(K); free(V); free(scores);
}

// Feed-forward network implementation
static void compute_ffn(float *x, const cortex_tensor *up, const cortex_tensor *gate, const cortex_tensor *down,
                       float *temp1, float *temp2, int n_embd, int n_ff, int layer_idx, int diag_i) {
    (void)diag_i;  // Avoid unused warning
    (void)layer_idx;  // Avoid unused warning
    
    // Up projection
    // Weight ne=[1024, 3072] means 3072 rows x 1024 columns, output 3072 dimensions
    if (up->type == GGML_TYPE_F32) {
        matmul_f32((float*)up->data, x, temp1, (int)up->ne[1], 1, (int)up->ne[0]);
    } else if (up->type == GGML_TYPE_Q8_0) {
        matmul_q8_0_f32(up->data, x, temp1, (int)up->ne[1], 1, (int)up->ne[0]);
    } else if (up->type == GGML_TYPE_Q4_0) {
        matmul_q4_0_f32(up->data, x, temp1, (int)up->ne[1], 1, (int)up->ne[0]);
    }
    
    // Gating
    if (gate->type == GGML_TYPE_F32) {
        matmul_f32((float*)gate->data, x, temp2, (int)gate->ne[1], 1, (int)gate->ne[0]);
    } else if (gate->type == GGML_TYPE_Q8_0) {
        // Fix: weight is ne[1] rows x ne[0] columns
        matmul_q8_0_f32(gate->data, x, temp2, (int)gate->ne[1], 1, (int)gate->ne[0]);
    } else if (gate->type == GGML_TYPE_Q4_0) {
        matmul_q4_0_f32(gate->data, x, temp2, (int)gate->ne[1], 1, (int)gate->ne[0]);
    }
    
    // SiLU activation
    silu(temp2, n_ff);
    
    // Element-wise multiplication
    for (int i = 0; i < n_ff; i++) {
        temp1[i] *= temp2[i];
    }
    
    // Down projection
    // Down weight ne=[n_ff, n_embd], for example [3072, 1024]
    if (down->type == GGML_TYPE_F32) {
        matmul_f32((float*)down->data, temp1, x, (int)down->ne[1], 1, (int)down->ne[0]);
    } else if (down->type == GGML_TYPE_Q8_0) {
        // Fix: weight is ne[1] rows x ne[0] columns
        matmul_q8_0_f32(down->data, temp1, x, (int)down->ne[1], 1, (int)down->ne[0]);
    } else if (down->type == GGML_TYPE_Q4_0) {
        matmul_q4_0_f32(down->data, temp1, x, (int)down->ne[1], 1, (int)down->ne[0]);
    }
}


// Create context
cortex_error cortex_context_new(cortex_model *model, cortex_context **ctx) {
    *ctx = malloc(sizeof(cortex_context));
    if (!*ctx) {
        return CORTEX_ERROR_MEMORY;
    }
    
    (*ctx)->model = model;
    (*ctx)->n_ctx_used = 0;
    
    int n_embd = model->hparams.n_embd;
    int n_ctx = model->hparams.n_ctx;
    int n_ff = model->hparams.n_ff;
    
    // Allocate computation buffers
    (*ctx)->embd = malloc(n_embd * sizeof(float));
    (*ctx)->attn_buf = malloc(n_embd * sizeof(float));
    (*ctx)->ffn_buf = malloc(n_ff * 2 * sizeof(float));
    (*ctx)->logits = malloc(model->hparams.n_vocab * sizeof(float));
    
    // Allocate KV cache
    (*ctx)->k_cache = malloc(model->hparams.n_layer * n_ctx * n_embd * sizeof(float));
    (*ctx)->v_cache = malloc(model->hparams.n_layer * n_ctx * n_embd * sizeof(float));
    
    if (!(*ctx)->embd || !(*ctx)->attn_buf || !(*ctx)->ffn_buf || 
        !(*ctx)->logits || !(*ctx)->k_cache || !(*ctx)->v_cache) {
        cortex_context_free(*ctx);
        return CORTEX_ERROR_MEMORY;
    }
    
    return CORTEX_OK;
}

// Free context
void cortex_context_free(cortex_context *ctx) {
    if (!ctx) return;
    
    free(ctx->embd);
    free(ctx->attn_buf);
    free(ctx->ffn_buf);
    free(ctx->logits);
    free(ctx->k_cache);
    free(ctx->v_cache);
    free(ctx);
}

// Inference function
cortex_error cortex_eval(cortex_context *ctx, const cortex_token *tokens, int n_tokens, cortex_pos n_past) {
    cortex_model *model = ctx->model;
    int n_embd = model->hparams.n_embd;
    int n_vocab = model->hparams.n_vocab;
    
    // Process each token
    for (int i = 0; i < n_tokens; i++) {
        cortex_token token = tokens[i];
        
        // Token embedding - Use correct nb[1] stride
        if (model->tok_embeddings->type == GGML_TYPE_F32) {
            float *embd_data = (float*)model->tok_embeddings->data;
            const uint64_t ne0 = model->tok_embeddings->ne[0];
            for (int j = 0; j < n_embd; j++) {
                ctx->embd[j] = embd_data[j + token * ne0];
            }
        } else if (model->tok_embeddings->type == GGML_TYPE_Q8_0) {
            // Use nb[1] as byte stride for each token
            const size_t byte_offset = token * model->tok_embeddings->nb[1];
            
            dequantize_q8_0((char*)model->tok_embeddings->data + byte_offset, ctx->embd, n_embd);
        }
        
        if (false && i == 0) {
            for (int j = 0; j < 10; j++) {
                printf("%.6f ", ctx->embd[j]);
            }
            printf("\n");
            
        }
        
        // Inference layer by layer
        for (uint32_t layer_idx = 0; layer_idx < model->hparams.n_layer; layer_idx++) {
            cortex_layer *layer = &model->layers[layer_idx];
            
            // Save residual
            float *residual = malloc(n_embd * sizeof(float));
            if (!residual) {
                return CORTEX_ERROR_MEMORY;
            }
            memcpy(residual, ctx->embd, n_embd * sizeof(float));
            
            // Attention normalization
            rms_norm(ctx->embd, (float*)layer->attn_norm->data, 
                     layer->attn_norm_b ? (float*)layer->attn_norm_b->data : NULL, 
                     n_embd, 1e-6f);
            // Compute Q, K, V (GQA: K and V output dimensions are n_embd_kv, Q output dimension is n_embd_q)
            int n_head_kv = model->hparams.n_head_kv;
            int head_dim = 128;  // Qwen3's head_dim is fixed at 128
            int n_embd_q = model->hparams.n_head * head_dim;  // Q's total dimension = 16 * 128 = 2048
            int n_embd_kv = n_head_kv * head_dim;              // K/V's total dimension = 8 * 128 = 1024
            
            float *q = malloc(n_embd_q * sizeof(float));  // Q: 2048 dimensions
            float *k = malloc(n_embd_kv * sizeof(float)); // K: 1024 dimensions
            float *v = malloc(n_embd_kv * sizeof(float)); // V: 1024 dimensions
            
            if (!q || !k || !v) {
                free(q); free(k); free(v);
                return CORTEX_ERROR_MEMORY;
            }
            
            // Q: wq[1024, 2048] @ embd[1024] -> q[2048]
            if (layer->wq->type == GGML_TYPE_F32) {
                matmul_f32((float*)layer->wq->data, ctx->embd, q, (int)layer->wq->ne[1], 1, (int)layer->wq->ne[0]);
            } else if (layer->wq->type == GGML_TYPE_Q8_0) {
                matmul_q8_0_f32(layer->wq->data, ctx->embd, q, (int)layer->wq->ne[1], 1, (int)layer->wq->ne[0]);
            }
            
            // K: wk[1024, 1024] @ embd[1024] -> k[1024]
            if (layer->wk->type == GGML_TYPE_F32) {
                matmul_f32((float*)layer->wk->data, ctx->embd, k, (int)layer->wk->ne[1], 1, (int)layer->wk->ne[0]);
            } else if (layer->wk->type == GGML_TYPE_Q8_0) {
                matmul_q8_0_f32(layer->wk->data, ctx->embd, k, (int)layer->wk->ne[1], 1, (int)layer->wk->ne[0]);
            }
            // V: wv[1024, 1024] @ embd[1024] -> v[1024]
            if (layer->wv->type == GGML_TYPE_F32) {
                matmul_f32((float*)layer->wv->data, ctx->embd, v, (int)layer->wv->ne[1], 1, (int)layer->wv->ne[0]);
            } else if (layer->wv->type == GGML_TYPE_Q8_0) {
                matmul_q8_0_f32(layer->wv->data, ctx->embd, v, (int)layer->wv->ne[1], 1, (int)layer->wv->ne[0]);
            }
            
        // Qwen3 specific: Q and K normalization (before RoPE, matching qwen3.c order)
        if (layer->attn_q_norm) {
            // Check type and convert to FP32 (if needed)
            float* q_norm_w = NULL;
            if (layer->attn_q_norm->type == GGML_TYPE_F32) {
                q_norm_w = (float*)layer->attn_q_norm->data;
            } else if (layer->attn_q_norm->type == GGML_TYPE_F16) {
                // Need to convert FP16 to FP32
                q_norm_w = malloc(head_dim * sizeof(float));
                uint16_t* fp16_data = (uint16_t*)layer->attn_q_norm->data;
                for (int j = 0; j < head_dim; j++) {
                    q_norm_w[j] = fp16_to_fp32(fp16_data[j]);
                }
            }
            
            
            // Q normalization: normalize each head separately (using RMS norm, not Layer norm!)
            if (q_norm_w) {
                for (int h = 0; h < (int)model->hparams.n_head; h++) {
                    // Use RMS norm: normalize first, then apply weights
                    float *q_head = q + h * head_dim;
                    
                    // Step 1: RMS normalize
                    double sum_sq = 0.0;
                    for (int d = 0; d < head_dim; d++) {
                        sum_sq += (double)(q_head[d] * q_head[d]);
                    }
                    float rms = sqrtf((float)(sum_sq / head_dim) + 1e-6f);
                    
                    // Step 2: Normalize and apply weights
                    for (int d = 0; d < head_dim; d++) {
                        q_head[d] = (q_head[d] / rms) * q_norm_w[d];
                    }
                }
                // If converted from FP16, free temporary buffer
                if (layer->attn_q_norm->type == GGML_TYPE_F16) {
                    free(q_norm_w);
                }
            }
        }
        if (layer->attn_k_norm) {
            // Check type and convert to FP32 (if needed)
            float* k_norm_w = NULL;
            if (layer->attn_k_norm->type == GGML_TYPE_F32) {
                k_norm_w = (float*)layer->attn_k_norm->data;
            } else if (layer->attn_k_norm->type == GGML_TYPE_F16) {
                // Need to convert FP16 to FP32
                k_norm_w = malloc(head_dim * sizeof(float));
                uint16_t* fp16_data = (uint16_t*)layer->attn_k_norm->data;
                for (int j = 0; j < head_dim; j++) {
                    k_norm_w[j] = fp16_to_fp32(fp16_data[j]);
                }
            }
            
            
            // K normalization: normalize each KV head separately (using RMS norm, not Layer norm!)
            if (k_norm_w) {
                for (int h = 0; h < (int)model->hparams.n_head_kv; h++) {
                    // Use RMS norm: normalize first, then apply weights
                    float *k_head = k + h * head_dim;
                    
                    // Step 1: RMS normalize
                    double sum_sq = 0.0;
                    for (int d = 0; d < head_dim; d++) {
                        sum_sq += (double)(k_head[d] * k_head[d]);
                    }
                    float rms = sqrtf((float)(sum_sq / head_dim) + 1e-6f);
                    
                    // Step 2: Normalize and apply weights
                    for (int d = 0; d < head_dim; d++) {
                        k_head[d] = (k_head[d] / rms) * k_norm_w[d];
                    }
                }
                // If converted from FP16, free temporary buffer
                if (layer->attn_k_norm->type == GGML_TYPE_F16) {
                    free(k_norm_w);
                }
            }
            }
            
            // Attention computation
            int n_head = model->hparams.n_head;
            float *attn_out = malloc(n_embd_q * sizeof(float));  // attn_out dimension = n_embd_q = 2048
            
            if (!attn_out) {
                free(q); free(k); free(v);
                return CORTEX_ERROR_MEMORY;
            }
            
            
            // Prepare normalization weights for Q and K
            float *q_norm_weight = NULL;
            float *k_norm_weight = NULL;
            
            if (layer->attn_q_norm) {
                q_norm_weight = (float*)layer->attn_q_norm->data;
            }
            if (layer->attn_k_norm) {
                k_norm_weight = (float*)layer->attn_k_norm->data;
            }
            
            compute_attention(q, k, v, attn_out, n_head, n_head_kv, head_dim, 
                             n_past + i + 1, n_past + i,
                             ctx->k_cache + layer_idx * model->hparams.n_ctx * n_embd_kv,
                             ctx->v_cache + layer_idx * model->hparams.n_ctx * n_embd_kv,
                             model->hparams.rope_freq_base, model->hparams.rope_freq_scale,
                             q_norm_weight, k_norm_weight);
            // Qwen3 specific: Apply RMS norm to attention output before wo projection
            // Match llama.cpp's build_norm logic: normalize first, then apply weights if available
            // Step 1: RMS normalize (without applying weights)
            rms_norm_only(attn_out, n_embd_q, 1e-6f);
            
            // Step 2: Apply weights if available
            if (layer->attn_sub_norm) {
                float *sub_norm_weight = NULL;
                if (layer->attn_sub_norm->type == GGML_TYPE_F16) {
                    sub_norm_weight = (float*)malloc(n_embd_q * sizeof(float));
                    for (int i = 0; i < n_embd_q; i++) {
                        sub_norm_weight[i] = fp16_to_fp32(((uint16_t*)layer->attn_sub_norm->data)[i]);
                    }
                } else {
                    sub_norm_weight = (float*)layer->attn_sub_norm->data;
                }
                
                // Apply weights
                for (int i = 0; i < n_embd_q; i++) {
                    attn_out[i] *= sub_norm_weight[i];
                }
                
                if (layer->attn_sub_norm->type == GGML_TYPE_F16) {
                    free(sub_norm_weight);
                }
            }
            
            
            // Output projection: Check actual ne values of wo
            float *attn_proj = malloc(n_embd * sizeof(float));
            if (!attn_proj) {
                free(q); free(k); free(v); free(attn_out);
                return CORTEX_ERROR_MEMORY;
            }
            
            
            // wo[2048, 1024] @ attn_out[2048] -> attn_proj[1024]
            if (layer->wo->type == GGML_TYPE_F32) {
                matmul_f32((float*)layer->wo->data, attn_out, attn_proj, (int)layer->wo->ne[1], 1, (int)layer->wo->ne[0]);
            } else if (layer->wo->type == GGML_TYPE_Q8_0) {
                matmul_q8_0_f32(layer->wo->data, attn_out, attn_proj, (int)layer->wo->ne[1], 1, (int)layer->wo->ne[0]);
            }
            
            // Residual connection
            for (int j = 0; j < n_embd; j++) {
                ctx->embd[j] = residual[j] + attn_proj[j];
            }
            free(residual);
            
            // FFN normalization - Save FFN residual
            float *ffn_residual = malloc(n_embd * sizeof(float));
            if (!ffn_residual) {
                free(q); free(k); free(v); free(attn_out); free(attn_proj);
                return CORTEX_ERROR_MEMORY;
            }
            memcpy(ffn_residual, ctx->embd, n_embd * sizeof(float));
            
            
            // Experiment: Check if weight clipping is needed
            
            rms_norm(ctx->embd, (float*)layer->ffn_norm->data,
                     layer->ffn_norm_b ? (float*)layer->ffn_norm_b->data : NULL,
                     n_embd, 1e-6f);
            
            // Feed-forward network
            compute_ffn(ctx->embd, layer->ffn_up, layer->ffn_gate, layer->ffn_down,
                        ctx->ffn_buf, ctx->ffn_buf + model->hparams.n_ff, n_embd, model->hparams.n_ff, layer_idx, i);
            
            // FFN residual connection
            for (int j = 0; j < n_embd; j++) {
                ctx->embd[j] = ffn_residual[j] + ctx->embd[j];
            }
            free(q); free(k); free(v); free(attn_out); free(attn_proj); free(ffn_residual);
        }
        
        if (false) {}
        
        // Output normalization (before output projection)
        rms_norm(ctx->embd, (float*)model->output_norm->data,
                 model->output_norm_b ? (float*)model->output_norm_b->data : NULL,
                 n_embd, 1e-6f);
        // Output projection: output[1024, 151936] @ embd[1024] -> logits[151936]
        if (model->output->type == GGML_TYPE_F32) {
            matmul_f32((float*)model->output->data, ctx->embd, ctx->logits, (int)model->output->ne[1], 1, (int)model->output->ne[0]);
        } else if (model->output->type == GGML_TYPE_Q8_0) {
            matmul_q8_0_f32(model->output->data, ctx->embd, ctx->logits, (int)model->output->ne[1], 1, (int)model->output->ne[0]);
        }
    }
    
    ctx->n_ctx_used += n_tokens;
    return CORTEX_OK;
}

// get logits
float* cortex_get_logits(cortex_context *ctx) {
    return ctx->logits;
}

// sample function
cortex_token cortex_sample(cortex_context *ctx, float temperature, float top_p __attribute__((unused)), int top_k) {
    int n_vocab = ctx->model->hparams.n_vocab;
    
    // Create logits copy to avoid modifying original logits
    float *logits = malloc(n_vocab * sizeof(float));
    memcpy(logits, ctx->logits, n_vocab * sizeof(float));
    
    
    // Apply temperature
    if (temperature > 0.0f) {
        for (int i = 0; i < n_vocab; i++) {
            logits[i] /= temperature;
        }
    }
    
    // Top-k sampling: Keep the top k largest logits
    if (top_k > 0 && top_k < n_vocab) {
        // Create index array
        int *indices = malloc(n_vocab * sizeof(int));
        for (int i = 0; i < n_vocab; i++) {
            indices[i] = i;
        }
        
        // Bubble sort to find top k maximum values (optimization: only partial sorting needed)
        for (int i = 0; i < top_k && i < n_vocab; i++) {
            for (int j = i + 1; j < n_vocab; j++) {
                if (logits[indices[j]] > logits[indices[i]]) {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }
        
        // Set logits not in top-k to negative infinity
        float threshold = (top_k < n_vocab) ? logits[indices[top_k - 1]] : -1e9f;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] < threshold) {
                logits[i] = -INFINITY;
            }
        }
        
        free(indices);
    }
    
    // Softmax
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    
    for (int i = 0; i < n_vocab; i++) {
        logits[i] /= sum;
    }
    
    // Random sampling
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    
    for (int i = 0; i < n_vocab; i++) {
        cumsum += logits[i];
        if (r <= cumsum) {
            free(logits);
            return i;
        }
    }
    
    free(logits);
    return n_vocab - 1;
}

