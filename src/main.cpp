/**
 * Qwen3 Inference Engine - Direct GGUF File Loading
 * 
 * This program implements a complete inference engine for Qwen3 models
 * stored in GGUF format, including tokenization, model forward pass,
 * and text generation capabilities.
 */

#include "ggml.h"
#include "gguf.h"
#include "gguf-loader.h"
#include "gguf_token.h"
#include "../utils/log.h"
#include "../utils/chat_template.h"
#include "../pkg/nlohmann/json.hpp"

#include <unordered_map>
#include <vector>
#include <cassert>
#include <climits>
#include <cstring>
#include <cstdarg>
#include <cinttypes>
#include <ctime>
#include <random>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdint>

// GGUF metadata keys
#define KV_GENERAL_ARCHITECTURE          "general.architecture"
#define KV_GENERAL_NAME                  "general.name"

#define KV_TOKENIZER_MODEL               "tokenizer.ggml.model"
#define KV_TOKENIZER_LIST                "tokenizer.ggml.tokens"
#define KV_TOKENIZER_TOKEN_TYPE          "tokenizer.ggml.token_type"
#define KV_TOKENIZER_SCORES              "tokenizer.ggml.scores"
#define KV_TOKENIZER_BOS_ID              "tokenizer.ggml.bos_token_id"
#define KV_TOKENIZER_EOS_ID              "tokenizer.ggml.eos_token_id"
#define KV_TOKENIZER_UNK_ID              "tokenizer.ggml.unknown_token_id"
#define KV_TOKENIZER_SEP_ID              "tokenizer.ggml.seperator_token_id"
#define KV_TOKENIZER_PAD_ID              "tokenizer.ggml.padding_token_id"
#define KV_TOKENIZER_HF_JSON             "tokenizer.huggingface.json"

// Tensor names in GGUF format
#define TN_TOKEN_EMBD  "token_embd.weight"
#define TN_OUTPUT_NORM "output_norm.weight"
#define TN_OUTPUT      "output.weight"
#define TN_ATTN_NORM   "blk.%d.attn_norm.weight"
#define TN_ATTN_Q      "blk.%d.attn_q.weight"
#define TN_ATTN_K      "blk.%d.attn_k.weight"
#define TN_ATTN_V      "blk.%d.attn_v.weight"
#define TN_ATTN_OUTPUT "blk.%d.attn_output.weight"
#define TN_FFN_NORM    "blk.%d.ffn_norm.weight"
#define TN_FFN_GATE    "blk.%d.ffn_gate.weight"
#define TN_FFN_DOWN    "blk.%d.ffn_down.weight"
#define TN_FFN_UP      "blk.%d.ffn_up.weight"

// ----------------------------------------------------------------------------
// Global Variables
// Global group size for quantization
int GS = 0;

// Maximum input prompt buffer size
#define PROMPT_BUFFER_SIZE 32768

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


// Cortex vocabulary for actual tokenization/decoding
static const struct cortex_vocab * g_cortex_vocab = NULL;

// ----------------------------------------------------------------------------
// Core Data Structures

/**
 * Model configuration parameters
 */
typedef struct {
    int dim;              // Transformer dimension
    int hidden_dim;       // Hidden dimension for FFN layers
    int n_layers;         // Number of transformer layers
    int n_heads;          // Number of query attention heads
    int n_kv_heads;       // Number of key/value heads (may be less than query heads for grouped-query attention)
    int vocab_size;       // Vocabulary size
    int seq_len;          // Maximum sequence length
    int head_dim;         // Dimension per attention head
    int shared_classifier; // If 1, classifier weights are shared with token embeddings
    int group_size;       // Quantization group size
    float rms_eps;        // RMS normalization epsilon value
} Config;

/**
 * Quantized tensor representation using group-wise quantization
 */
typedef struct {
    int8_t *q;    // Quantized values
    float *s;     // Scaling factors (one per group)
} QuantizedTensor;

/**
 * Transformer model weights
 * All weights are stored in quantized format (Q8_0) except normalization layers
 */
typedef struct {
    // Token embedding weights
    QuantizedTensor *q_tokens;              // Quantized token embeddings (vocab_size, dim)
    float *token_embedding_table;           // Dequantized token embeddings (vocab_size, dim)
    
    // RMS normalization weights
    float *rms_att_weight;                  // Attention RMSNorm weights (n_layers, dim)
    float *rms_ffn_weight;                   // FFN RMSNorm weights (n_layers, dim)
    float *rms_final_weight;                // Final layer RMSNorm weights (dim,)
    
    // Attention weights
    QuantizedTensor *wq;                    // Query projection (n_layers, dim, n_heads * head_dim)
    QuantizedTensor *wk;                    // Key projection (n_layers, dim, n_kv_heads * head_dim)
    QuantizedTensor *wv;                    // Value projection (n_layers, dim, n_kv_heads * head_dim)
    QuantizedTensor *wo;                    // Output projection (n_layers, n_heads * head_dim, dim)
    
    // Qwen3-specific: QK-RMSNorm weights
    float *q_norm_weights;                  // Query RMSNorm weights (n_layers, head_dim)
    float *k_norm_weights;                   // Key RMSNorm weights (n_layers, head_dim)
    
    // Feed-forward network weights
    QuantizedTensor *w1;                    // FFN gate weights (n_layers, hidden_dim, dim)
    QuantizedTensor *w2;                    // FFN down projection (n_layers, dim, hidden_dim)
    QuantizedTensor *w3;                    // FFN up projection (n_layers, hidden_dim, dim)
    
    // Classifier weights (optional - may be shared with token embeddings)
    QuantizedTensor *wcls;                  // Output classifier weights (vocab_size, dim)
} TransformerWeights;

/**
 * Runtime state for transformer inference
 * Contains activation buffers and KV cache for efficient generation
 */
typedef struct {
    // Current activations
    float *x;                               // Current token activation (dim,)
    float *xb;                              // Residual branch buffer (dim,)
    float *hb;                              // FFN hidden buffer (hidden_dim,)
    float *hb2;                             // FFN second buffer (hidden_dim,)
    
    // Quantized activation buffers
    QuantizedTensor xq;                     // Quantized x (dim,)
    QuantizedTensor hq;                     // Quantized hb (hidden_dim,)
    
    // Attention computation buffers
    float *q;                               // Query vector (dim,)
    float *k;                               // Key vector (dim,)
    float *v;                               // Value vector (dim,)
    float *att;                             // Attention scores (n_heads, seq_len)
    
    // Output
    float *logits;                          // Final output logits (vocab_size,)
    
    // KV cache for efficient generation
    float *key_cache;                       // Cached keys (n_layers, seq_len, kv_dim)
    float *value_cache;                     // Cached values (n_layers, seq_len, kv_dim)
} RunState;

// ----------------------------------------------------------------------------
// Quantization Definitions

// Q8_0 quantization: 32 int8 values per block with one FP16 scale
#define QK8_0 32
typedef uint16_t ggml_half;
typedef struct {
    ggml_half d;       // Scale factor (FP16)
    int8_t qs[QK8_0];  // Quantized values (32 elements per block)
} block_q8_0;


/**
 * Complete transformer model
 */
typedef struct {
    Config config;              // Model hyperparameters
    TransformerWeights weights; // Model weights
    RunState state;             // Runtime activation buffers
    float *data;                // Memory-mapped data pointer (if used)
    ssize_t file_size;          // Checkpoint file size in bytes
} Transformer;

// Global: Last loaded configuration (used for RMSNorm epsilon)
Config g_last_config;


// ----------------------------------------------------------------------------
// Helper Functions

/**
 * Get a uint32 value from GGUF context with qwen3 prefix
 */
static uint32_t get_u32_arch_c(const struct gguf_context *c, const char *suffix) {
    char key[128];
    snprintf(key, sizeof(key), "qwen3.%s", suffix);
    
    int k = gguf_find_key(c, key);
    if (k < 0) {
        fprintf(stderr, "Missing key for %s\n", suffix);
        return (uint32_t)0;
    }
    
    return gguf_get_val_u32(c, k);
}


/**
 * Get reference logits from cortex model (for validation/testing)
 * Loads full model and returns logits for the last position
 */
static bool get_reference_logits(const char * gguf_file, const char * rendered_prompt, std::vector<float> & out_logits) {
    struct cortex_model_params mp = cortex_model_default_params();
    mp.vocab_only = false;
    mp.use_mmap = true;
    mp.check_tensors = false;
    struct cortex_model * model = cortex_model_load_from_file(gguf_file, mp);
    if (!model) return false;
    
    const struct cortex_vocab * vocab = cortex_model_get_vocab(model);
    struct cortex_context_params cp = cortex_context_default_params();
    cp.n_ctx = 512;
    struct cortex_context * ctx = cortex_new_context_with_model(model, cp);
    if (!ctx) { cortex_model_free(model); return false; }

    // Tokenize input
    std::vector<cortex_token> toks(4096);
    int32_t nt = cortex_tokenize(vocab, rendered_prompt, (int)strlen(rendered_prompt), 
                                  toks.data(), (int)toks.size(), false, true);
    if (nt < 0) {
        nt = -nt;
    }
    toks.resize(nt);

    // Run forward pass
    cortex_batch batch = {};
    std::vector<cortex_token> tks = toks;
    std::vector<cortex_pos>   pos(tks.size());
    std::vector<int8_t>      logits_mask(tks.size(), 0);
    for (int i = 0; i < (int)tks.size(); ++i) pos[i] = i;
    batch.token  = tks.data();
    batch.pos    = pos.data();
    batch.logits = logits_mask.data();
    batch.n_tokens = (int)tks.size();
    if (cortex_decode(ctx, batch) != 0) { 
        cortex_free(ctx); 
        cortex_model_free(model); 
        return false; 
    }

    // Extract logits from last position
    float * ref = cortex_get_logits(ctx);
    int n_vocab = cortex_vocab_n_tokens(vocab);
    out_logits.assign(ref, ref + n_vocab);

    cortex_free(ctx);
    cortex_model_free(model);
    return true;
}

// ----------------------------------------------------------------------------
// Memory Management

/**
 * Allocate runtime state buffers for transformer inference
 */
void malloc_run_state(RunState* s, Config *p) {
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    // Activation buffers
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(all_heads_dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    
    // Quantized activation buffers
    s->xq = (QuantizedTensor) { 
        .q = (int8_t*)calloc(all_heads_dim, sizeof(int8_t)), 
        .s = (float*)calloc(all_heads_dim / GS, sizeof(float)) 
    };
    s->hq = (QuantizedTensor) { 
        .q = (int8_t*)calloc(p->hidden_dim, sizeof(int8_t)), 
        .s = (float*)calloc(p->hidden_dim / GS, sizeof(float)) 
    };
    
    // Attention buffers
    s->q = (float*)calloc(all_heads_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    
    // Output buffer
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    
    // KV cache for all layers
    s->key_cache = (float*)calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    // Verify all allocations succeeded
    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Neural Network Building Blocks

/**
 * RMSNorm normalization
 * Computes: output = weight * (x / sqrt(mean(x^2) + eps))
 */
void rmsnorm(float *o, float *x, float *weight, int size) {
    // Calculate sum of squares
    float ss = 0;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];

    // Normalization factor
    ss = 1.0f / sqrtf((ss / size) + 1e-6f);

    // Normalize and scale
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

/**
 * Softmax activation function
 * Applies softmax in-place: exp(x - max) / sum(exp(x - max))
 */
void softmax(float *x, int size) {
    // Find max value for numerical stability
    float max_val = 0;
    for (int i = 0; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];

    // Compute exp(x - max) and sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

// ----------------------------------------------------------------------------
// Quantization Functions

/**
 * Dequantize a tensor: convert from int8 + scale back to float32
 */
void dequantize(QuantizedTensor *qx, float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = qx->q[i] * qx->s[i / GS];
}

/**
 * Quantize a tensor: convert from float32 to int8 with per-group scaling
 * Uses group-wise quantization with GS elements per group
 */
void quantize(QuantizedTensor *qx, float *x, int n) {
    for (int group = 0; group < n / GS; group++) {
        // Find the max absolute value in the current group
        float wmax = 0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
                wmax = val;
        }

        // Calculate scaling factor to fit into int8 range [-127, 127]
        float scale = wmax / 127.0f;
        qx->s[group] = scale;

        // Quantize values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            int8_t quantized = (int8_t) round(quant_value);
            qx->q[group * GS + i] = quantized;
        }
    }
}

/**
 * Allocate and initialize an array of QuantizedTensor structures
 */
QuantizedTensor* alloc_quantized_tensors(int n_layers, int size_each) {
    QuantizedTensor* tensors = (QuantizedTensor*)malloc(n_layers * sizeof(QuantizedTensor));
    if (!tensors) return NULL;
    
    for (int i = 0; i < n_layers; i++) {
        tensors[i].q = (int8_t*)malloc(size_each * sizeof(int8_t));
        tensors[i].s = (float*)malloc((size_each / GS) * sizeof(float));
        if (!tensors[i].q || !tensors[i].s) {
            // Clean up allocated memory on failure
            for (int j = 0; j <= i; j++) {
                free(tensors[j].q);
                free(tensors[j].s);
            }
            free(tensors);
            return NULL;
        }
    }
    return tensors;
}

/**
 * Load Q8_0 format data into QuantizedTensor structure
 * Converts from GGML Q8_0 block format to our internal format
 */
void load_q8_0_to_quantized(QuantizedTensor* qt, const void* q8_0_data, int size) {
    const block_q8_0* blocks = (const block_q8_0*)q8_0_data;
    int num_blocks = size / QK8_0;
    
    for (int block = 0; block < num_blocks; block++) {
        // Convert FP16 scale to FP32
        float scale = ggml_fp16_to_fp32(blocks[block].d);
        qt->s[block] = scale;
        
        // Copy quantized values
        for (int i = 0; i < QK8_0; i++) {
            qt->q[block * QK8_0 + i] = blocks[block].qs[i];
        }
    }
}

/**
 * Quantized matrix-vector multiplication
 * Computes: output = W @ x where both W and x are quantized
 * This is the most compute-intensive operation in the forward pass
 * 
 * @param xout Output vector (d,)
 * @param x Input vector quantized (n,)
 * @param w Weight matrix quantized (d, n)
 * @param n Input dimension
 * @param d Output dimension
 */
void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // Uses group-wise quantization with GS elements per group
    
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        // Compute dot product in groups of GS for efficient quantization
        for (int j = 0; j <= n - GS; j += GS) {
            // Integer dot product within group
            int32_t ival = 0;
            for (int k = 0; k < GS; k++)
                ival += x->q[j + k] * w->q[in + j + k];

            // Apply scaling factors
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = val;
    }
}

/**
 * Forward pass through the transformer
 * Processes a single token at position pos and returns logits for next token
 * 
 * @param transformer The transformer model
 * @param token Current token ID
 * @param pos Position in the sequence
 * @return Pointer to logits array (vocab_size,)
 */
float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // KV sharing multiplier for grouped-query attention
    int all_heads_dim = p->n_heads * p->head_dim;

    // Embed current token
    memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

    // Forward through all transformer layers
    for (int l = 0; l < p->n_layers; l++) {
        // Set pointers to KV cache slots for this layer and position
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // Attention RMSNorm
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * p->dim, p->dim);

        // Compute Q, K, V projections
        quantize(&s->xq, s->xb, p->dim);
        matmul(s->q, &s->xq, w->wq + l, p->dim, all_heads_dim);
        matmul(s->k, &s->xq, w->wk + l, p->dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, p->dim, kv_dim);

        // Qwen3-specific: Q-RMSNorm + RoPE (Rotary Position Embedding) per query head
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;

            // Apply RMSNorm to query head
            rmsnorm(q, q, w->q_norm_weights + l * p->head_dim, p->head_dim);
            
            // Apply RoPE rotation (treating head_dim/2 pairs as complex numbers)
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq);
                float sin_freq = sinf(pos * freq);

                float x = q[j];                           // Real part
                float y = q[j + p->head_dim/2];          // Imaginary part

                // Complex rotation: (x + iy) * (cos + i*sin)
                q[j] = x * cos_freq - y * sin_freq;      // New real
                q[j + p->head_dim/2] = x * sin_freq + y * cos_freq; // New imag
            }
        }

        // K-RMSNorm + RoPE per key head
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * p->head_dim;

            // Apply RMSNorm to key head
            rmsnorm(k, k, w->k_norm_weights + l * p->head_dim, p->head_dim);
            
            // Apply RoPE rotation
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq);
                float sin_freq = sinf(pos * freq);

                float x = k[j];
                float y = k[j + p->head_dim/2];

                k[j] = x * cos_freq - y * sin_freq;
                k[j + p->head_dim/2] = x * sin_freq + y * cos_freq;
            }
        }

        // Multi-head attention
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;
            float *att = s->att + h * p->seq_len;
            
            // Compute attention scores with all previous positions
            for (int t = 0; t <= pos; t++) {
                // Get key vector (with KV sharing for grouped-query attention)
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                
                // Dot product attention score
                float score = 0;
                for (int i = 0; i < p->head_dim; i++)
                    score += q[i] * k[i];

                // Scale by sqrt(head_dim)
                att[t] = score / sqrtf(p->head_dim);
            }

            // Softmax to get attention weights
            softmax(att, pos + 1);

            // Weighted sum of values
            float *xb = s->xb + h * p->head_dim;
            memset(xb, 0, p->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                for (int i = 0; i < p->head_dim; i++)
                    xb[i] += att[t] * v[i];
            }
        }

        // Output projection
        quantize(&s->xq, s->xb, all_heads_dim);
        matmul(s->xb, &s->xq, w->wo + l, all_heads_dim, p->dim);

        // Residual connection
        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];

        // FFN: SwiGLU activation
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * p->dim, p->dim);

        // Compute w1(x) and w3(x) for SwiGLU: w2(SiLU(w1(x)) * w3(x))
        quantize(&s->xq, s->xb, p->dim);
        matmul(s->hb, &s->xq, w->w1 + l, p->dim, p->hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, p->dim, p->hidden_dim);

        // SwiGLU: hb = SiLU(hb) * hb2, where SiLU(x) = x * sigmoid(x)
        for (int i = 0; i < p->hidden_dim; i++)
            s->hb[i] *= s->hb2[i] * (1.0f / (1.0f + expf(-s->hb[i])));

        // Final FFN projection
        quantize(&s->hq, s->hb, p->hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, p->hidden_dim, p->dim);

        // Residual connection
        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];
    }

    // Final layer normalization
    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    // Project to vocabulary logits
    quantize(&s->xq, s->x, p->dim);
    matmul(s->logits, &s->xq, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}
// ----------------------------------------------------------------------------
// Tokenizer Implementation
// Tokenizer structure is defined in gguf_token.h

/**
 * Convert a byte (0-255) to its Unicode representation (GPT-2 BPE style)
 */
static char* byte_to_unicode_char(uint8_t byte, char* buffer) {
    static uint32_t byte_to_unicode[256];
    static int initialized = 0;
    
    if (!initialized) {
        // Build GPT-2 style bytes_to_unicode mapping
        // Initialize all to 0 (unset)
        for (int i = 0; i < 256; i++) {
            byte_to_unicode[i] = 0;
        }
        
        // Printable ASCII (0x21-0x7E) map to themselves
        for (int i = 0x21; i <= 0x7E; i++) {
            byte_to_unicode[i] = i;
        }
        // Extended Latin (0xA1-0xAC) map to themselves
        for (int i = 0xA1; i <= 0xAC; i++) {
            byte_to_unicode[i] = i;
        }
        // More extended (0xAE-0xFF) map to themselves
        for (int i = 0xAE; i <= 0xFF; i++) {
            byte_to_unicode[i] = i;
        }
        
        // Non-printable/unmapped bytes map to 256+n
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if (byte_to_unicode[i] == 0) {
                byte_to_unicode[i] = 256 + n;
                n++;
            }
        }
        initialized = 1;
    }
    
    uint32_t unicode = byte_to_unicode[byte];
    // Convert Unicode codepoint to UTF-8
    if (unicode < 0x80) {
        buffer[0] = (char)unicode;
        buffer[1] = 0;
    } else if (unicode < 0x800) {
        buffer[0] = (char)(0xC0 | (unicode >> 6));
        buffer[1] = (char)(0x80 | (unicode & 0x3F));
        buffer[2] = 0;
    } else if (unicode < 0x10000) {
        buffer[0] = (char)(0xE0 | (unicode >> 12));
        buffer[1] = (char)(0x80 | ((unicode >> 6) & 0x3F));
        buffer[2] = (char)(0x80 | (unicode & 0x3F));
        buffer[3] = 0;
    } else {
        buffer[0] = (char)(0xF0 | (unicode >> 18));
        buffer[1] = (char)(0x80 | ((unicode >> 12) & 0x3F));
        buffer[2] = (char)(0x80 | ((unicode >> 6) & 0x3F));
        buffer[3] = (char)(0x80 | (unicode & 0x3F));
        buffer[4] = 0;
    }
    return buffer;
}

/**
 * Convert Unicode UTF-8 string back to byte (for BPE tokenizer)
 */
static uint8_t unicode_to_byte(const char* utf8_str) {
    static std::unordered_map<std::string, uint8_t> utf8_to_byte_map;
    static int initialized = 0;
    
    if (!initialized) {
        // Build reverse mapping
        for (int i = 0x21; i <= 0x7E; i++) {
            char buf[5];
            byte_to_unicode_char(i, buf);
            utf8_to_byte_map[std::string(buf)] = i;
        }
        for (int i = 0xA1; i <= 0xAC; i++) {
            char buf[5];
            byte_to_unicode_char(i, buf);
            utf8_to_byte_map[std::string(buf)] = i;
        }
        for (int i = 0xAE; i <= 0xFF; i++) {
            char buf[5];
            byte_to_unicode_char(i, buf);
            utf8_to_byte_map[std::string(buf)] = i;
        }
        int n = 0;
        for (int i = 0; i < 256; i++) {
            char buf[5];
            byte_to_unicode_char(i, buf);
            if (utf8_to_byte_map.find(std::string(buf)) == utf8_to_byte_map.end()) {
                utf8_to_byte_map[std::string(buf)] = i;
            }
        }
        initialized = 1;
    }
    
    std::string key(utf8_str);
    auto it = utf8_to_byte_map.find(key);
    if (it != utf8_to_byte_map.end()) {
        return it->second;
    }
    // If not found, try to return first byte (fallback)
    return utf8_str[0] & 0xFF;
}

/**
 * Decode token with Unicode-to-byte conversion
 */
static char* decode_token_bytes(Tokenizer *t, int token, char* output_buffer, size_t buffer_size) {
    if (token < 0 || token >= t->vocab_size || !t->vocab[token]) {
        output_buffer[0] = 0;
        return output_buffer;
    }
    
    const char* token_str = t->vocab[token];
    size_t len = strlen(token_str);
    size_t out_pos = 0;
    
    // Process each Unicode character and convert to byte
    size_t i = 0;
    while (i < len && out_pos < buffer_size - 1) {
        // Determine UTF-8 character length
        int utf8_len = 1;
        if ((unsigned char)token_str[i] >= 0xF0) {
            utf8_len = 4;
        } else if ((unsigned char)token_str[i] >= 0xE0) {
            utf8_len = 3;
        } else if ((unsigned char)token_str[i] >= 0xC0) {
            utf8_len = 2;
        }
        
        // Extract UTF-8 character
        char utf8_char[5] = {0};
        for (int j = 0; j < utf8_len && (i + j) < len; j++) {
            utf8_char[j] = token_str[i + j];
        }
        
        // Convert Unicode character to byte
        uint8_t byte = unicode_to_byte(utf8_char);
        output_buffer[out_pos++] = (char)byte;
        
        i += utf8_len;
    }
    output_buffer[out_pos] = 0;
    return output_buffer;
}

/**
 * Decode a single token to string
 */
char *decode(Tokenizer *t, int token) {
    static char decode_buffer[4096];
    return decode_token_bytes(t, token, decode_buffer, sizeof(decode_buffer));
}

/**
 * Look up string in vocabulary
 * @return Token index or -1 if not found
 */
int str_lookup(char *str, char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;
    return -1;
}

/**
 * Encode text into tokens using BPE tokenization
 * Implements byte-pair encoding with merge operations
 */
void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // Buffer for merge candidates (two consecutive tokens concatenated)
    // *2 for concat, +1 for null terminator, +2 for UTF8 safety
    char *str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char unicode_buffer[8];
    char special_token[64 + 1];

    *n_tokens = 0;

    // Process input byte-by-byte, converting to Unicode representation
    for (char *c = text; *c != 0; c++) {
        int id, found_special_token = 0;

        // Convert byte to Unicode representation (GPT-2 BPE style)
        uint8_t byte = (uint8_t)*c;
        byte_to_unicode_char(byte, unicode_buffer);

        // Handle special tokens (e.g., <|im_start|>, <|im_end|>)
        // Special tokens begin with < and end with >
        if (*c == '<') {
          int end_of_token_pos = -1;
          found_special_token = 0;
          for (int k = 0; *c != 0 && k < 64; k++) {
              if (c[k] == '>') {
                  end_of_token_pos = k;
                  break;
              }
          }

          if (end_of_token_pos != -1) {
              strncpy(special_token, c, end_of_token_pos + 1);
              special_token[end_of_token_pos + 1] = 0;

              id = str_lookup(special_token, t->vocab, t->vocab_size);
              if (id != -1) {
                  c += end_of_token_pos;
                  found_special_token = 1;
              }
          }
        }

        // Not a special token, look up Unicode representation in vocab
        if (!found_special_token) {
            id = str_lookup(unicode_buffer, t->vocab, t->vocab_size);
        }

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            printf("Warning: unknown character code point %d (Unicode: %s) in input, skipping.\n", byte, unicode_buffer);
            (*n_tokens)++;
        }
    }

    // BPE merge phase: iteratively merge best consecutive pairs
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        // Find the best merge candidate
        for (int i = 0; i < (*n_tokens - 1); i++) {
            // Concatenate consecutive tokens
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            // Select merge with highest score
            if (id != -1 && t->merge_scores[id] > best_score) {
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
            break; // No more merges possible

        // Merge the pair
        tokens[best_idx] = best_id;
        // Remove second token and shift remaining tokens
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];

        (*n_tokens)--;
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler Implementation

/**
 * Probability-index pair for top-p sampling
 */
typedef struct {
    float prob;
    int index;
} ProbIndex;

/**
 * Sampler configuration and state
 */
typedef struct {
    int vocab_size;
    ProbIndex *probindex;  // Buffer for top-p sampling
    float temperature;     // Temperature for sampling (0 = greedy)
    float topp;            // Top-p (nucleus) sampling threshold
    unsigned long long rng_state; // RNG state
} Sampler;

/**
 * Greedy sampling: return index with highest probability
 */
int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

/**
 * Multinomial sampling: sample from probability distribution
 * @param coin Random number in [0, 1) for CDF sampling
 */
int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1; // Fallback for rounding errors
}

/**
 * Comparison function for sorting probabilities (descending)
 */
int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

/**
 * Top-p (nucleus) sampling
 * Samples from the smallest set of tokens whose cumulative probability >= topp
 * This avoids sampling very low-probability tokens that could cause incoherent output
 */
int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {

    // Filter candidates: remove tokens with probability < (1-topp)/(n-1)
    // This is an optimization - these tokens cannot be in the final set
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    
    // Sort candidates by probability (descending)
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // Truncate to smallest set with cumulative probability >= topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    // Sample from truncated distribution
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index; // Fallback for rounding errors
}

/**
 * Initialize sampler with configuration
 */
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}


void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
        strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
        strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load prompt template %s\n", prompt_path); exit(EXIT_FAILURE); }
    fread(out_template, 1024, 1, file);
    fclose(file);
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char *)malloc(1);
            t->vocab[i][0] = 0; // add the string terminating token
        } else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char *)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = 0; // add the string terminating token
        }
    }
    fclose(file);

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void free_tokenizer(Tokenizer *t) {
    if (t->template_handle) {
        chat_template_free(t->template_handle);
        t->template_handle = NULL;
    }
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->merge_scores);
}
/**
 * Xorshift RNG: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
 */
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/**
 * Generate random float in [0, 1)
 */
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

/**
 * Sample next token from logits
 * Supports greedy (temperature=0), multinomial, and top-p sampling
 */
int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0) {
        // Greedy: take highest probability token
        return sample_argmax(logits, sampler->vocab_size);
    } else {
        // Apply temperature scaling
        for (int q = 0; q < sampler->vocab_size; q++) { 
            logits[q] /= sampler->temperature; 
        }
        
        // Convert logits to probabilities
        softmax(logits, sampler->vocab_size);
        
        // Sample with random seed
        float coin = random_f32(&sampler->rng_state);
        
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // Standard multinomial sampling
            return sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // Top-p (nucleus) sampling
            return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
}

/**
 * Free all memory allocated for transformer model
 */
void free_transformer(Transformer *t) {
    free(t->weights.token_embedding_table);
    free(t->weights.rms_att_weight);
    free(t->weights.rms_ffn_weight);
    
    // Free QuantizedTensor arrays
    if (t->weights.wq) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.wq[i].q);
            free(t->weights.wq[i].s);
        }
        free(t->weights.wq);
    }
    
    if (t->weights.wk) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.wk[i].q);
            free(t->weights.wk[i].s);
        }
        free(t->weights.wk);
    }
    
    if (t->weights.wv) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.wv[i].q);
            free(t->weights.wv[i].s);
        }
        free(t->weights.wv);
    }
    
    if (t->weights.wo) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.wo[i].q);
            free(t->weights.wo[i].s);
        }
        free(t->weights.wo);
    }
    
    if (t->weights.w1) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.w1[i].q);
            free(t->weights.w1[i].s);
        }
        free(t->weights.w1);
    }
    
    if (t->weights.w2) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.w2[i].q);
            free(t->weights.w2[i].s);
        }
        free(t->weights.w2);
    }
    
    if (t->weights.w3) {
        for (int i = 0; i < t->config.n_layers; i++) {
            free(t->weights.w3[i].q);
            free(t->weights.w3[i].s);
        }
        free(t->weights.w3);
    }
    
    // Special handling: wcls may be shared with token embeddings
    if (t->weights.wcls && t->config.shared_classifier == 0) {
        free(t->weights.wcls[0].q);
        free(t->weights.wcls[0].s);
        free(t->weights.wcls);
    }
    
    free(t->weights.q_norm_weights);
    free(t->weights.k_norm_weights);
    free(t->weights.rms_final_weight);
    
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// GGUF Loader Implementation

/**
 * Convert GGML tensor to float array (handles quantization)
 * Note: This function has limited conversion for large tensors
 */
static void convert_weights_gg_to_float(const struct ggml_tensor * gg_weights, float * float_weights) {
    if (!gg_weights || !float_weights) {
        LOG_ERR("%s: Invalid input parameters\n", __func__);
        return;
    }
    
    int size = 1;
    for (int dim = 0; dim < ggml_n_dims(gg_weights); ++dim) {
        size *= gg_weights->ne[dim];
    }
    
    //LOG_INF("%s: Converting tensor type %d, size %d, data: %p\n", __func__, gg_weights->type, size, gg_weights->data);
    
    // Handle different tensor types (including quantized)
    if (gg_weights->type == GGML_TYPE_F32) {
        // Direct copy for float32
        if (gg_weights->data) {
            memcpy(float_weights, gg_weights->data, size * sizeof(float));
        }
    } else {
        // For quantized types, try direct memory access
        if (gg_weights->data) {
            // Limit the conversion to avoid crashes
            int max_elements = size;
            if (max_elements > 1000000) {
                max_elements = 1000000; // Limit to 1M elements for testing
                LOG_INF("%s: Limiting conversion to %d elements for testing\n", __func__, max_elements);
            }
            
            // Try direct memory access for Q8_0
            if (gg_weights->type == GGML_TYPE_Q8_0) {
                LOG_INF("%s: Using direct memory access for Q8_0\n", __func__);
                // Q8_0 format: each block has 32 elements + 1 scale factor
                const int block_size = 32;
                const int n_blocks = (max_elements + block_size - 1) / block_size;
                
                for (int block = 0; block < n_blocks; block++) {
                    const uint8_t * block_data = (const uint8_t *)gg_weights->data + block * (block_size + sizeof(float));
                    const float scale = *(const float *)block_data;
                    const uint8_t * qdata = block_data + sizeof(float);
                    
                    int start_idx = block * block_size;
                    int end_idx = std::min(start_idx + block_size, max_elements);
                    
                    for (int i = start_idx; i < end_idx; i++) {
                        float_weights[i] = scale * (qdata[i - start_idx] - 128);
                    }
                }
            } else {
                // Fallback to element-wise conversion
                for (int i = 0; i < max_elements; i++) {
                    float_weights[i] = ggml_get_f32_1d(gg_weights, i);
                }
            }
            
            // Fill remaining with zeros
            for (int i = max_elements; i < size; i++) {
                float_weights[i] = 0.0f;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Dequantization Implementation

/**
 * Convert FP16 to FP32
 * Handles special cases: zero, denormal, infinity, NaN
 */
static float fp16_to_fp32(ggml_half h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    
    if (exp == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // Denormal number
            return (sign ? -1.0f : 1.0f) * (float)mantissa / (1 << 10) / (1 << 14);
        }
    } else if (exp == 31) {
        // Infinity or NaN
        return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    } else {
        // Normal number
        union { uint32_t i; float f; } u;
        u.i = (sign << 31) | ((exp + 112) << 23) | (mantissa << 13);
        return u.f;
    }
}

/**
 * Dequantize Q8_0 blocks to float array
 */
void dequantize_q8_0(const void* src, float* dst, int n_elements) {
    const block_q8_0* blocks = (const block_q8_0*)src;
    int n_blocks = n_elements / QK8_0;
    for (int i = 0; i < n_blocks; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK8_0; j++) 
            dst[i * QK8_0 + j] = blocks[i].qs[j] * d;
    }
}

/**
 * Read 1D GGML tensor to contiguous float array
 */
static void tensor_read_1d(struct ggml_tensor * t, float * out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) 
        out[i] = ggml_get_f32_nd(t, i, 0, 0, 0);
}

/**
 * Read 2D GGML tensor to row-major float array
 * GGML tensors: ne[0] is column (input dim), ne[1] is row (output dim)
 */
static void tensor_read_2d_rowmajor(struct ggml_tensor * t, float * out, int64_t rows, int64_t cols) {
    // out[r*cols + c] = t[c, r]
    for (int64_t r = 0; r < rows; ++r) {
        float * prow = out + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            prow[c] = ggml_get_f32_nd(t, c, r, 0, 0);
        }
    }
}


static float cosine_similarity(const float * a, const float * b, int n) {
    double da=0, db=0, dab=0; for (int i=0;i<n;++i){ double xa=a[i], xb=b[i]; da+=xa*xa; db+=xb*xb; dab+=xa*xb; }
    if (da==0||db==0) return 0.0f;
    return (float)(dab / (sqrt(da)*sqrt(db)));
}

// ----------------------------------------------------------------------------
// GGUF Loading Implementation

/**
 * Initialize quantized tensors from GGUF Q8_0 format
 * Converts from GGML block format to internal QuantizedTensor format
 */
QuantizedTensor *init_quantized_tensors_from_gguf(void *ptr, int n, int n_elements) {
    
    QuantizedTensor *res = (QuantizedTensor*) malloc(n * sizeof(QuantizedTensor));
    
    const block_q8_0* blocks = (const block_q8_0*)ptr;
    int n_blocks = n_elements / QK8_0;

    for (int i = 0; i < n; i++) {
        int8_t* q_data = (int8_t*) malloc(n_elements * sizeof(int8_t));
        float* s_data = (float*) malloc(n_blocks * sizeof(float));
        
        for (int j = 0; j < n_blocks; j++) {
            float d = fp16_to_fp32(blocks[j].d);
            s_data[j] = d;
            memcpy(q_data + j * QK8_0, blocks[j].qs, QK8_0 * sizeof(int8_t));
        }
        res[i].s = s_data;
        res[i].q = q_data;
    }
    return res;
}

/**
 * Initialize quantized tensors from memory-mapped data
 * Maps quantized values and scale factors from contiguous memory
 */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    QuantizedTensor *res = (QuantizedTensor*) malloc(n * sizeof(QuantizedTensor));

    for (int i = 0; i < n; i++) {
        // map quantized int8 values
        res[i].q = (int8_t*)*ptr;
        *ptr = (int8_t*)*ptr + size_each;
        // map scale factors
        res[i].s = (float*)*ptr;
        *ptr = (float*)*ptr + size_each / GS;
    }
    return res;
}


/**
 * Debug function: print first 16 elements of each weight tensor
 */
void print_weights_head(TransformerWeights *w, Config *p) {
    int n = 16;

    if (w->q_tokens && w->q_tokens->q && w->q_tokens->s) {
        printf("q_tokens[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->q_tokens[0].q[i]);
        }
        printf("q_tokens[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->q_tokens[0].s[i]);
        }
    }
    
    // Print first 16 elements of token_embedding_table
    printf("token_embedding_table[0..%d]:\n", n-1);
    for (int i = 0; i < n; i++) {
        printf("  [%d] = %f\n", i, w->token_embedding_table[i]);
    }

    // Print first 16 elements of rms_att_weight
    printf("rms_att_weight[0..%d]:\n", n-1);
    for (int i = 0; i < n; i++) {
        printf("  [%d] = %f\n", i, w->rms_att_weight[i]);
    }

    // Print first 16 elements of rms_ffn_weight
    printf("rms_ffn_weight[0..%d]:\n", n-1);
    for (int i = 0; i < n; i++) {
        printf("  [%d] = %f\n", i, w->rms_ffn_weight[i]);
    }

    // Print first 16 elements of q_norm_weights
    printf("q_norm_weights[0..%d]:\n", n-1);
    for (int i = 0; i < n; i++) {
        printf("  [%d] = %f\n", i, w->q_norm_weights[i]);
    }

    // Print first 16 elements of k_norm_weights
    printf("k_norm_weights[0..%d]:\n", n-1);
    for (int i = 0; i < n; i++) {
        printf("  [%d] = %f\n", i, w->k_norm_weights[i]);
    }

    // Print first 16 elements of rms_final_weight
    printf("rms_final_weight[0..%d]:\n", n-1);
    for (int i = 0; i < n; i++) {
        printf("  [%d] = %f\n", i, w->rms_final_weight[i]);
    }

    // Print first 16 elements of wcls (if exists)
    if (w->wcls && w->wcls->q && w->wcls->s) {
        printf("wcls.q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->wcls->q[i]);
        }
        printf("wcls.s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->wcls->s[i]);
        }
    }

    // Print first 16 elements of wq layer 0 (if exists)
    if (w->wq && w->wq->q && w->wq->s) {
        printf("wq[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->wq[0].q[i]);
        }
        printf("wq[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->wq[0].s[i]);
        }
    }

    // Print first 16 elements of wk layer 0 (if exists)
    if (w->wk && w->wk->q && w->wk->s) {
        printf("wk[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->wk[0].q[i]);
        }
        printf("wk[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->wk[0].s[i]);
        }
    }

    // Print first 16 elements of wv layer 0 (if exists)
    if (w->wv && w->wv->q && w->wv->s) {
        printf("wv[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->wv[0].q[i]);
        }
        printf("wv[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->wv[0].s[i]);
        }
    }

    // Print first 16 elements of wo layer 0 (if exists)
    if (w->wo && w->wo->q && w->wo->s) {
        printf("wo[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->wo[0].q[i]);
        }
        printf("wo[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->wo[0].s[i]);
        }
    }

    // Print first 16 elements of w1 layer 0 (if exists)
    if (w->w1 && w->w1->q && w->w1->s) {
        printf("w1[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->w1[0].q[i]);
        }
        printf("w1[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->w1[0].s[i]);
        }
    }

    // Print first 16 elements of w2 layer 0 (if exists)
    if (w->w2 && w->w2->q && w->w2->s) {
        printf("w2[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->w2[0].q[i]);
        }
        printf("w2[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->w2[0].s[i]);
        }
    }

    // Print first 16 elements of w3 layer 0 (if exists)
    if (w->w3 && w->w3->q && w->w3->s) {
        printf("w3[0].q[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %d\n", i, w->w3[0].q[i]);
        }
        printf("w3[0].s[0..%d]:\n", n-1);
        for (int i = 0; i < n; i++) {
            printf("  [%d] = %f\n", i, w->w3[0].s[i]);
        }
    }
}

/**
 * Load Qwen3 model from GGUF file
 * Loads architecture, weights, and tokenizer from GGUF format
 * 
 * @return 0 on success, 1 on failure
 */
int load_model_from_gguf(const char *filename, Config *config, TransformerWeights *weights, Tokenizer *tokenizer) {
    printf("Loading GGUF model from %s\n", filename);
    
    // Initialize GGUF context
    struct ggml_context * ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(filename, params);
    if (ctx == NULL) {
        fprintf(stderr, "Failed to load GGUF file %s\n", filename);
        return 1;
    }
    
    // Validate architecture
    const int arch_idx = gguf_find_key(ctx, "general.architecture");
    if (arch_idx < 0) {
        fprintf(stderr, "Missing architecture in GGUF file\n");
        gguf_free(ctx);
        return 1;
    }
    
    const char* arch = gguf_get_val_str(ctx, arch_idx);
    if (strcmp(arch, "qwen3") != 0) {
        fprintf(stderr, "Unsupported architecture: %s (supported: 'qwen3')\n", arch);
        gguf_free(ctx);
        return 1;
    }

    // Extract hyperparameters
    config->dim        = get_u32_arch_c(ctx, "embedding_length");
    config->hidden_dim = get_u32_arch_c(ctx, "feed_forward_length");
    config->n_layers   = get_u32_arch_c(ctx, "block_count");
    config->n_heads    = get_u32_arch_c(ctx, "attention.head_count");
    config->n_kv_heads = get_u32_arch_c(ctx, "attention.head_count_kv");
    config->seq_len    = get_u32_arch_c(ctx, "context_length");
    // epsilon / rope base
    {
        char key_eps[64]; snprintf(key_eps, sizeof(key_eps), "%s", "qwen3.attention.layer_norm_rms_epsilon");
        int ke = gguf_find_key(ctx, key_eps);
        config->rms_eps = ke >= 0 ? gguf_get_val_f32(ctx, ke) : 1e-6f;
    }
    
    // Get vocabulary size
    const int token_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (token_idx >= 0) {
        config->vocab_size = gguf_get_arr_n(ctx, token_idx);
    } else {
        fprintf(stderr, "Missing tokenizer list in GGUF file\n");
        gguf_free(ctx);
        return 1;
    }
    
    // Get head_dim: prefer qwen3.attention.key_length, fallback to dim / n_heads
    uint32_t key_len = get_u32_arch_c(ctx, "attention.key_length");
    if (key_len > 0) {
        config->head_dim = (int) key_len;
    }
    else {
        config->head_dim = config->dim / config->n_heads;
    }

    config->group_size = QK8_0; 
    GS = config->group_size;
    
    /**
     * Weights
     */
    //const int all_heads_dim = config->n_heads * config->head_dim;
    //const int kv_cols_total = config->n_kv_heads * config->head_dim;
    //printf("Loading %d layers, all_heads_dim=%d, kv_cols_total=%d\n", config->n_layers, all_heads_dim, kv_cols_total);
    char tensor_name[256];
    weights->rms_att_weight = (float*) malloc(config->n_layers * config->dim * sizeof(float)); 
    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_norm.weight", i);
        const int attn_norm_idx = gguf_find_tensor(ctx, tensor_name);
        if (attn_norm_idx >= 0) {
            struct ggml_tensor * attn_norm = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, attn_norm_idx));
            memcpy(weights->rms_att_weight + i * config->dim, attn_norm->data, config->dim * sizeof(float));
        } else {
            printf("Warning: Missing attn_norm.weight for layer %d\n", i);
            exit(1);
        }
    }

    weights->rms_ffn_weight = (float*) malloc(config->n_layers * config->dim * sizeof(float));
    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_norm.weight", i);
        const int ffn_norm_idx = gguf_find_tensor(ctx, tensor_name);
        if (ffn_norm_idx >= 0) {
            struct ggml_tensor * ffn_norm = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, ffn_norm_idx));
            memcpy(weights->rms_ffn_weight + i * config->dim, ffn_norm->data, config->dim * sizeof(float));
        } else {
            printf("Warning: Missing ffn_norm.weight for layer %d\n", i);
            exit(1);
        }
    }


    weights->rms_final_weight = (float*) malloc(config->dim * sizeof(float));
    const int norm_idx = gguf_find_tensor(ctx, "output_norm.weight");
    if (norm_idx >= 0) {
        struct ggml_tensor * norm = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, norm_idx));
        memcpy(weights->rms_final_weight, norm->data, config->dim * sizeof(float));
    } else {
        printf("Warning: Missing output_norm.weight for layer %d\n", config->n_layers - 1);
        exit(1);    
    }

    weights->q_norm_weights = (float*) malloc(config->n_layers * config->head_dim * sizeof(float));
    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_q_norm.weight", i);
        const int qn_idx = gguf_find_tensor(ctx, tensor_name);
        if (qn_idx >= 0) {
            struct ggml_tensor * qn = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, qn_idx));
            memcpy(weights->q_norm_weights + i * config->head_dim, qn->data, config->head_dim * sizeof(float));
        } else {
            printf("Warning: Missing attn_q_norm.weight for layer %d\n", i);
            exit(1);
        }
    }

    weights->k_norm_weights = (float*) malloc(config->n_layers * config->head_dim * sizeof(float));
    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_k_norm.weight", i);
        const int kn_idx = gguf_find_tensor(ctx, tensor_name);
        if (kn_idx >= 0) {
            struct ggml_tensor * kn = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, kn_idx));
            memcpy(weights->k_norm_weights + i * config->head_dim, kn->data, config->head_dim * sizeof(float));
        } else {
            printf("Warning: Missing attn_k_norm.weight for layer %d\n", i);
            exit(1);
        }
    }


    // Load token embeddings
    const int tok_embd_idx = gguf_find_tensor(ctx, "token_embd.weight");
    if (tok_embd_idx >= 0) {
        struct ggml_tensor * tok_embd = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, tok_embd_idx));
        if (tok_embd->type == GGML_TYPE_Q8_0) {
            weights->q_tokens = init_quantized_tensors_from_gguf((void*)tok_embd->data, 1,config->vocab_size * config->dim);
            weights->token_embedding_table = (float*) malloc(config->vocab_size * config->dim * sizeof(float));
            dequantize(weights->q_tokens, weights->token_embedding_table, config->vocab_size * config->dim);
        } else {
            
            fprintf(stderr, "tok_embd->type != GGML_TYPE_Q8_0\n");
            exit(1);
        }
    }
    else {
        fprintf(stderr, "Missing token embeddings in GGUF file\n");
        exit(1);
    }
    
    weights->wq = alloc_quantized_tensors(config->n_layers, config->dim * config->n_heads * config->head_dim);
    weights->wk = alloc_quantized_tensors(config->n_layers, config->dim * config->n_kv_heads * config->head_dim);
    weights->wv = alloc_quantized_tensors(config->n_layers, config->dim * config->n_kv_heads * config->head_dim);
    weights->wo = alloc_quantized_tensors(config->n_layers, config->n_heads * config->head_dim * config->dim);
    weights->w1 = alloc_quantized_tensors(config->n_layers, config->hidden_dim * config->dim);
    weights->w2 = alloc_quantized_tensors(config->n_layers, config->dim * config->hidden_dim);
    weights->w3 = alloc_quantized_tensors(config->n_layers, config->hidden_dim * config->dim);
    weights->wcls = alloc_quantized_tensors(1, config->vocab_size * config->dim);

    for (int i = 0; i < config->n_layers; i++) { 
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_q.weight", i);
        const int wq_idx = gguf_find_tensor(ctx, tensor_name);
        if (wq_idx >= 0) {
            struct ggml_tensor * wq = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, wq_idx));
           
            if (wq->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->wq[i], wq->data, config->dim * config->n_heads * config->head_dim);
            } else {
                fprintf(stderr, "Warning: attn_q.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing attn_q.weight for layer %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_k.weight", i);
        const int wk_idx = gguf_find_tensor(ctx, tensor_name);
        if (wk_idx >= 0) {
            struct ggml_tensor * wk = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, wk_idx));
            if (wk->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->wk[i], wk->data, config->dim * config->n_kv_heads * config->head_dim);
            } else {
                fprintf(stderr, "Warning: attn_k.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing attn_k.weight for layer %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_v.weight", i);
        const int wv_idx = gguf_find_tensor(ctx, tensor_name);
        if (wv_idx >= 0) {
            struct ggml_tensor * wv = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, wv_idx));
            if (wv->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->wv[i], wv->data, config->dim * config->n_kv_heads * config->head_dim);
            } else {
                fprintf(stderr, "Warning: attn_v.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing attn_v.weight for layer %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.attn_output.weight", i);
        const int wo_idx = gguf_find_tensor(ctx, tensor_name);
        if (wo_idx >= 0) {
            struct ggml_tensor * wo = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, wo_idx));
            if (wo->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->wo[i], wo->data, config->n_heads * config->head_dim * config->dim);
            } else {
                fprintf(stderr, "Warning: attn_output.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing attn_output.weight for layer %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_gate.weight", i);
        const int w1_idx = gguf_find_tensor(ctx, tensor_name);
        if (w1_idx >= 0) {
            struct ggml_tensor * w1 = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, w1_idx));     
            if (w1->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->w1[i], w1->data, config->hidden_dim * config->dim);
            } else {
                fprintf(stderr, "Warning: ffn_gate.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing ffn_gate.weight for layer %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_down.weight", i);
        const int w2_idx = gguf_find_tensor(ctx, tensor_name);
        if (w2_idx >= 0) {
            struct ggml_tensor * w2 = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, w2_idx));
            if (w2->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->w2[i], w2->data, config->dim * config->hidden_dim);
            } else {
                fprintf(stderr, "Warning: ffn_down.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing ffn_down.weight for layer %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < config->n_layers; i++) {
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_up.weight", i);
        const int w3_idx = gguf_find_tensor(ctx, tensor_name);
        if (w3_idx >= 0) {
            struct ggml_tensor * w3 = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, w3_idx));
            if (w3->type == GGML_TYPE_Q8_0) {
                load_q8_0_to_quantized(&weights->w3[i], w3->data, config->hidden_dim * config->dim);
            } else {
                fprintf(stderr, "Warning: ffn_up.weight for layer %d is not Q8_0\n", i);
                exit(1);
            }
        } else {
            fprintf(stderr, "Warning: Missing ffn_up.weight for layer %d\n", i);
            exit(1);
        }
    }
    
      
    // Load output classifier weights (or mark as shared with token embeddings)
    const int output_idx = gguf_find_tensor(ctx, "output.weight");
    if (output_idx >= 0) {
        struct ggml_tensor * output = ggml_get_tensor(ctx_data, gguf_get_tensor_name(ctx, output_idx));
        if (output->type == GGML_TYPE_Q8_0) {
            load_q8_0_to_quantized(&weights->wcls[0], output->data, config->vocab_size * config->dim);
        } else {
            fprintf(stderr, "Warning: output.weight is not Q8_0\n");
            exit(1);
        }
        config->shared_classifier = 0;
    } else {
        // Shared classifier weights - requires special handling
        // Free wcls QuantizedTensor memory
        free(weights->wcls[0].q);
        free(weights->wcls[0].s);
        free(weights->wcls);
        weights->wcls = NULL; // Mark as shared
        config->shared_classifier = 1;
    }
    
    // Load tokenizer from GGUF file
    printf("Loading tokenizer from GGUF\n");
    if (load_tokenizer_from_gguf(ctx, tokenizer, 0)) {
        fprintf(stderr, "Failed to load tokenizer from GGUF file\n");
        ggml_free(ctx_data);
        gguf_free(ctx);
        return 1;
    }
    
    // Initialize cortex tokenizer (vocab only) for actual tokenization/decoding
    cortex_backend_init();
    struct cortex_model_params lparams = cortex_model_default_params();
    lparams.vocab_only = true;
    lparams.use_mmap   = true;
    lparams.check_tensors = false;
    struct cortex_model * tmp_model = cortex_model_load_from_file(filename, lparams);
    if (tmp_model) {
        g_cortex_vocab = cortex_model_get_vocab(tmp_model);
        // Note: model not freed - vocab pointer depends on its lifetime (lightweight vocab_only mode)
    } else {
        fprintf(stderr, "Warning: failed to init cortex vocab, fallback to simplified tokenizer.\n");
    }

    ggml_free(ctx_data);
    gguf_free(ctx);
    
    printf("Successfully loaded GGUF model\n");
    return 0;
}


// ----------------------------------------------------------------------------
// Helper Functions: Chat Template Rendering

static void render_prompt_with_template(
    Tokenizer *tokenizer,
    const char *user_prompt,
    const char *system_prompt,
    bool add_generation_prompt,
    char *output_buffer,
    size_t buffer_size)
{
    if (tokenizer->use_chat_template && tokenizer->template_handle) {
        // Build JSON using nlohmann/json
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        
        if (system_prompt && strlen(system_prompt) > 0) {
            nlohmann::ordered_json sys_msg = nlohmann::ordered_json::object();
            sys_msg["role"] = "system";
            sys_msg["content"] = system_prompt;
            messages.push_back(sys_msg);
        }
        
        if (user_prompt && strlen(user_prompt) > 0) {
            nlohmann::ordered_json user_msg = nlohmann::ordered_json::object();
            user_msg["role"] = "user";
            user_msg["content"] = user_prompt;
            messages.push_back(user_msg);
        }
        
        std::string messages_json = messages.dump();
        
        // Call template renderer
        char* result = chat_template_render(
            tokenizer->template_handle,
            messages_json.c_str(),
            add_generation_prompt,
            output_buffer,
            buffer_size
        );
        
        if (!result) {
            // Template rendering failed, fall back to sprintf
            fprintf(stderr, "Warning: Template render failed, falling back to sprintf\n");
            if (system_prompt) {
                snprintf(output_buffer, buffer_size, tokenizer->system_prompt_template, system_prompt, user_prompt);
            } else {
                snprintf(output_buffer, buffer_size, tokenizer->prompt_template, user_prompt);
            }
        }
    } else {
        // Use old sprintf fallback
        if (system_prompt) {
            snprintf(output_buffer, buffer_size, tokenizer->system_prompt_template, system_prompt, user_prompt);
        } else {
            snprintf(output_buffer, buffer_size, tokenizer->prompt_template, user_prompt);
        }
    }
}

// ----------------------------------------------------------------------------
// Text Generation Loop

/**
 * Generate text from prompt
 * Runs forward pass iteratively, sampling tokens until EOS or max length
 */
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, char *system_prompt) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // Render user/system prompts into Qwen3 prompt template format
    char rendered_prompt[PROMPT_BUFFER_SIZE];
    render_prompt_with_template(tokenizer, prompt, system_prompt, true, rendered_prompt, sizeof(rendered_prompt));

    // Encode prompt string into token sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc((strlen(rendered_prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);
    }


    // Generation loop
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < transformer->config.seq_len) {
        // Forward pass: get logits for next token
        float *logits = forward(transformer, token, pos);

        // Determine next token
        if (pos < num_prompt_tokens - 1) {
            // Still processing prompt: use next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // Generate mode: sample from logits
            next = sample(sampler, logits);
        }
        pos++;

        // Decode and print token
        printf("%s", decode(tokenizer, token));
        fflush(stdout);
        token = next;

        // Stop on EOS or BOS token (after prompt)
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id))
            break;
    }
    printf("\n");
    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
            buffer[len - 1] = 0; // strip newline
    }
}

// ----------------------------------------------------------------------------
// Chat Loop Implementation

/**
 * Interactive chat loop
 * Alternates between user input and model generation
 */
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt) {
    char user_prompt[PROMPT_BUFFER_SIZE];
    char rendered_prompt[PROMPT_BUFFER_SIZE];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(PROMPT_BUFFER_SIZE * sizeof(int));

    int user_turn = 1; // User starts
    int next;
    int token;
    int pos = 0;

    while (1) {
        // Check context window limit
        if (pos >= transformer->config.seq_len) {
            printf("\n(context window full, clearing)\n");
            user_turn = 1;
            pos = 0;
        }

        // User turn: get input
        if (user_turn) {
            if (cli_user_prompt != NULL) {
                // Use command-line prompt (only at start)
                if (pos > 0)
                    break;
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // Read from stdin
                read_stdin("\n> ", user_prompt, sizeof(user_prompt));
                if (!user_prompt[0])
                    break;
            }

            // Render prompt with template (include system prompt only at start)
            if (pos == 0 && system_prompt) {
                render_prompt_with_template(tokenizer, user_prompt, system_prompt, true, rendered_prompt, sizeof(rendered_prompt));
            } else {
                render_prompt_with_template(tokenizer, user_prompt, NULL, true, rendered_prompt, sizeof(rendered_prompt));
            }
            
            // Encode prompt to tokens
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            pos = 0;
            user_turn = 0;
        }

        // Determine input token
        if (pos < num_prompt_tokens) {
            token = prompt_tokens[pos];
        } else {
            token = next;
        }

        // Forward pass and sampling
        float *logits = forward(transformer, token, pos++);
        next = sample(sampler, logits);

        // Assistant response generation
        if (pos >= num_prompt_tokens) {
            if (token == tokenizer->bos_token_id || token == tokenizer->eos_token_id) {
                // End of assistant turn
                printf("\n");
                user_turn = 1;
            } else if (next != tokenizer->bos_token_id && next != tokenizer->eos_token_id) {
                // Print generated token
                printf("%s", decode(tokenizer, next));
                fflush(stdout);
            }
        }
    }
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// Command Line Interface

void error_usage() {
    fprintf(stderr, "Usage:   cortexllm <checkpoint> [options]\n");
    fprintf(stderr, "Example: cortexllm Qwen3-0.6B-Q8_0.gguf -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    fprintf(stderr, "  -T <string> override chat template from GGUF with custom template\n");
    fprintf(stderr, "  -j <int>    jinja template mode: 0 = disable, 1 = enable (default: auto)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // Initialize logging system (optional)
    // common_log_init();
    
    // default parameters
    char *gguf_file = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "chat";        // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int enable_thinking = 0;    // 1 enables thinking
    int ctx_length = 0;         // context length
    char *custom_template = NULL; // custom template override
    int jinja_mode = -1;        // -1 = auto, 0 = disable, 1 = enable

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { 
        gguf_file = argv[1];
    } 
    else { 
        error_usage(); 
    }

  
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'T') { custom_template = argv[i + 1]; }
        else if (argv[i][1] == 'j') { jinja_mode = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;


   // Load model
   Config config;
   TransformerWeights weights;
   Tokenizer tokenizer;
   
   if (load_model_from_gguf(gguf_file, &config, &weights, &tokenizer)) {
       fprintf(stderr, "Failed to load GGUF model\n");
       return 1;
   }
   
   // Apply command line template overrides
   if (custom_template) {
       // Free old template (if exists)
       if (tokenizer.template_handle) {
           chat_template_free(tokenizer.template_handle);
           tokenizer.template_handle = NULL;
       }
       
       // Get BOS/EOS token strings
       const char* bos_token_str = NULL;
       const char* eos_token_str = NULL;
       if (tokenizer.bos_token_id < (unsigned int)tokenizer.vocab_size && tokenizer.vocab[tokenizer.bos_token_id]) {
           bos_token_str = tokenizer.vocab[tokenizer.bos_token_id];
       }
       if (tokenizer.eos_token_id < (unsigned int)tokenizer.vocab_size && tokenizer.vocab[tokenizer.eos_token_id]) {
           eos_token_str = tokenizer.vocab[tokenizer.eos_token_id];
       }
       
       // Load custom template
       tokenizer.template_handle = chat_template_load_from_string(custom_template, bos_token_str, eos_token_str);
       if (tokenizer.template_handle) {
           tokenizer.use_chat_template = true;
           printf("Loaded custom template from command line\n");
       } else {
           fprintf(stderr, "Warning: Failed to load custom template, using default\n");
           tokenizer.use_chat_template = false;
       }
   }
   
   // Apply jinja mode override
   if (jinja_mode == 0) {
       // Disable Jinja template
       if (tokenizer.template_handle) {
           chat_template_free(tokenizer.template_handle);
           tokenizer.template_handle = NULL;
       }
       tokenizer.use_chat_template = false;
       printf("Jinja template disabled by command line\n");
   } else if (jinja_mode == 1) {
       // Force enable Jinja template (if not already loaded)
       if (!tokenizer.template_handle && !tokenizer.use_chat_template) {
           // Try loading from prompt_template string
           if (strlen(tokenizer.prompt_template) > 0) {
               const char* bos_token_str = NULL;
               const char* eos_token_str = NULL;
               if (tokenizer.bos_token_id < (unsigned int)tokenizer.vocab_size && tokenizer.vocab[tokenizer.bos_token_id]) {
                   bos_token_str = tokenizer.vocab[tokenizer.bos_token_id];
               }
               if (tokenizer.eos_token_id < (unsigned int)tokenizer.vocab_size && tokenizer.vocab[tokenizer.eos_token_id]) {
                   eos_token_str = tokenizer.vocab[tokenizer.eos_token_id];
               }
               tokenizer.template_handle = chat_template_load_from_string(tokenizer.prompt_template, bos_token_str, eos_token_str);
               if (tokenizer.template_handle) {
                   tokenizer.use_chat_template = true;
                   printf("Force enabled Jinja template from prompt_template\n");
               }
           }
       }
   }

   // Tokenizer already loaded from GGUF file
   
   // Set sequence length
   if (ctx_length != 0 && ctx_length <= config.seq_len)
       config.seq_len = ctx_length;

   // Build transformer
   Transformer transformer;
   transformer.config = config;
   g_last_config = config;
   transformer.weights = weights;
   transformer.data = NULL;
   transformer.file_size = 0;
   
   // Allocate runtime state
   malloc_run_state(&transformer.state, &transformer.config);

   // Build sampler
   Sampler sampler;
   build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);


    if (!prompt)
        printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n", transformer.config.dim, transformer.config.hidden_dim, transformer.config.n_layers, transformer.config.n_heads, transformer.config.n_kv_heads, transformer.config.head_dim, transformer.config.seq_len, transformer.config.vocab_size, transformer.config.shared_classifier, transformer.config.group_size);

    if (strcmp(mode, "generate") == 0) {
        printf("Prompt: %s\n", prompt ? prompt : "NULL");
        generate(&transformer, &tokenizer, &sampler, prompt, system_prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
