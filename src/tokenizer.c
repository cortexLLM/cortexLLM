#include "cortex_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

// Simple BPE tokenizer implementation
typedef struct {
    char *text;
    float score;
} bpe_token;

// ============================================================================
// BPE byte decoding implementation - reference llama.cpp/src/unicode.cpp
// ============================================================================

// UTF-8 helper functions
static inline uint32_t utf8_len(uint8_t src) {
    const uint32_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return lookup[src >> 4];
}

// Decode Unicode code point from UTF-8 byte sequence
static uint32_t utf8_to_codepoint(const char *utf8, size_t *len) {
    uint8_t first = (uint8_t)utf8[0];
    
    if (first < 0x80) {  // 1-byte
        *len = 1;
        return first;
    } else if ((first & 0xE0) == 0xC0) {  // 2-byte
        *len = 2;
        return ((first & 0x1F) << 6) | ((uint8_t)utf8[1] & 0x3F);
    } else if ((first & 0xF0) == 0xE0) {  // 3-byte
        *len = 3;
        return ((first & 0x0F) << 12) | (((uint8_t)utf8[1] & 0x3F) << 6) | ((uint8_t)utf8[2] & 0x3F);
    } else if ((first & 0xF8) == 0xF0) {  // 4-byte
        *len = 4;
        return ((first & 0x07) << 18) | (((uint8_t)utf8[1] & 0x3F) << 12) | 
               (((uint8_t)utf8[2] & 0x3F) << 6) | ((uint8_t)utf8[3] & 0x3F);
    }
    
    *len = 1;
    return 0xFFFD;  // replacement character
}

// BPE byte to UTF-8 mapping table (reference llama.cpp/src/unicode.cpp)
// This mapping ensures all 256 byte values can be represented with UTF-8 characters
typedef struct {
    uint32_t codepoint;  // Unicode code point
    uint8_t byte_value;  // corresponding byte value
} byte_mapping;

// Build UTF-8 to byte reverse mapping table
static byte_mapping utf8_to_byte_table[256 + 256];  // maximum 512 mappings
static int utf8_to_byte_table_size = 0;
static bool utf8_to_byte_table_initialized = false;

static void init_utf8_to_byte_table() {
    if (utf8_to_byte_table_initialized) return;
    
    utf8_to_byte_table_size = 0;
    
    // 1. Printable ASCII: 0x21-0x7E (i.e., '!' to '~')
    for (int ch = 0x21; ch <= 0x7E; ch++) {
        utf8_to_byte_table[utf8_to_byte_table_size].codepoint = ch;
        utf8_to_byte_table[utf8_to_byte_table_size].byte_value = ch;
        utf8_to_byte_table_size++;
    }
    
    // 2. Extended ASCII: 0xA1-0xAC
    for (int ch = 0xA1; ch <= 0xAC; ch++) {
        utf8_to_byte_table[utf8_to_byte_table_size].codepoint = ch;
        utf8_to_byte_table[utf8_to_byte_table_size].byte_value = ch;
        utf8_to_byte_table_size++;
    }
    
    // 3. Extended ASCII: 0xAE-0xFF
    for (int ch = 0xAE; ch <= 0xFF; ch++) {
        utf8_to_byte_table[utf8_to_byte_table_size].codepoint = ch;
        utf8_to_byte_table[utf8_to_byte_table_size].byte_value = ch;
        utf8_to_byte_table_size++;
    }
    
    // 4. Non-printable characters mapped to 256+n
    int n = 0;
    for (int ch = 0; ch < 256; ch++) {
        // Check if already mapped
        bool already_mapped = false;
        for (int i = 0; i < utf8_to_byte_table_size; i++) {
            if (utf8_to_byte_table[i].byte_value == ch) {
                already_mapped = true;
                break;
            }
        }
        
        if (!already_mapped) {
            utf8_to_byte_table[utf8_to_byte_table_size].codepoint = 256 + n;
            utf8_to_byte_table[utf8_to_byte_table_size].byte_value = ch;
            utf8_to_byte_table_size++;
            n++;
        }
    }
    
    utf8_to_byte_table_initialized = true;
}

// Find corresponding byte value based on Unicode code point
static int codepoint_to_byte(uint32_t codepoint) {
    if (!utf8_to_byte_table_initialized) {
        init_utf8_to_byte_table();
    }
    
    for (int i = 0; i < utf8_to_byte_table_size; i++) {
        if (utf8_to_byte_table[i].codepoint == codepoint) {
            return utf8_to_byte_table[i].byte_value;
        }
    }
    
    return -1;  // Not found
}

// Find corresponding Unicode code point based on byte value
static uint32_t byte_to_codepoint(uint8_t byte) {
    if (!utf8_to_byte_table_initialized) {
        init_utf8_to_byte_table();
    }
    
    for (int i = 0; i < utf8_to_byte_table_size; i++) {
        if (utf8_to_byte_table[i].byte_value == byte) {
            return utf8_to_byte_table[i].codepoint;
        }
    }
    
    return 0xFFFD;  // Not found, return replacement character
}

// Encode byte as UTF-8 string (first step of BPE encoding)
static char *encode_byte_to_utf8(uint8_t byte) {
    uint32_t codepoint = byte_to_codepoint(byte);
    
    // Convert to UTF-8 based on code point
    static char buf[5];
    
    if (codepoint < 0x80) {
        buf[0] = (char)codepoint;
        buf[1] = '\0';
    } else if (codepoint < 0x800) {
        buf[0] = (char)(0xC0 | (codepoint >> 6));
        buf[1] = (char)(0x80 | (codepoint & 0x3F));
        buf[2] = '\0';
    } else if (codepoint < 0x10000) {
        buf[0] = (char)(0xE0 | (codepoint >> 12));
        buf[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (codepoint & 0x3F));
        buf[3] = '\0';
    } else {
        buf[0] = (char)(0xF0 | (codepoint >> 18));
        buf[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (codepoint & 0x3F));
        buf[4] = '\0';
    }
    
    return buf;
}

// BPE decoding function - decode UTF-8 encoded token string to original bytes
static char *decode_bpe_token(const char *token_text) {
    if (!token_text) return NULL;
    
    init_utf8_to_byte_table();
    
    size_t token_len = strlen(token_text);
    // Allocate enough space (worst case: each character corresponds to one byte)
    char *decoded = malloc(token_len + 1);
    if (!decoded) return NULL;
    
    int decoded_len = 0;
    size_t i = 0;
    
    while (i < token_len) {
        size_t utf8_char_len;
        uint32_t codepoint = utf8_to_codepoint(token_text + i, &utf8_char_len);
        
        int byte_val = codepoint_to_byte(codepoint);
        if (byte_val >= 0) {
            decoded[decoded_len++] = (char)byte_val;
        } else {
            // If mapping not found, keep original character (handle special tokens)
            for (size_t j = 0; j < utf8_char_len && (i + j) < token_len; j++) {
                decoded[decoded_len++] = token_text[i + j];
            }
        }
        
        i += utf8_char_len;
    }
    
    decoded[decoded_len] = '\0';
    return decoded;
}

// Encode text as BPE UTF-8 string
static char *encode_text_to_bpe_utf8(const char *text) {
    if (!text) return NULL;
    
    init_utf8_to_byte_table();
    
    size_t text_len = strlen(text);
    // Allocate enough space (worst case: each byte corresponds to 4 UTF-8 bytes)
    char *encoded = malloc(text_len * 4 + 1);
    if (!encoded) return NULL;
    
    int encoded_len = 0;
    
    for (size_t i = 0; i < text_len; i++) {
        const char *utf8_char = encode_byte_to_utf8((uint8_t)text[i]);
        int utf8_len = strlen(utf8_char);
        memcpy(encoded + encoded_len, utf8_char, utf8_len);
        encoded_len += utf8_len;
    }
    
    encoded[encoded_len] = '\0';
    return encoded;
}

// Find token index in vocabulary
static int find_token_index(cortex_model *model, const char *text) {
    for (uint32_t i = 0; i < model->vocab.size; i++) {
        if (model->vocab.tokens[i] && strcmp(model->vocab.tokens[i], text) == 0) {
            return i;
        }
    }
    return -1;
}

// BPE tokenization implementation - improved version: using greedy longest match algorithm
cortex_error cortex_tokenize(cortex_model *model, const char *text, cortex_token *tokens, int max_tokens, int *n_tokens) {
    if (!model || !text || !tokens || !n_tokens) {
        return CORTEX_ERROR_INVALID_PARAM;
    }
    
    *n_tokens = 0;
    
    // Step 1: Convert text to BPE UTF-8 encoding
    char *encoded = encode_text_to_bpe_utf8(text);
    if (!encoded) {
        return CORTEX_ERROR_MEMORY;
    }
    
    size_t encoded_len = strlen(encoded);
    size_t pos = 0;
    
    // Step 2: Use greedy longest match algorithm to find tokens
    while (pos < encoded_len && *n_tokens < max_tokens) {
        int best_token_id = -1;
        size_t best_token_len = 0;
        
        // Try to match the longest token from current position
        // Start from the longest possible length (limited to 256 bytes)
        size_t max_try_len = encoded_len - pos;
        if (max_try_len > 256) max_try_len = 256;
        
        for (size_t try_len = max_try_len; try_len > 0; try_len--) {
            // Extract substring
            char *substr = malloc(try_len + 1);
            if (!substr) {
                free(encoded);
                return CORTEX_ERROR_MEMORY;
            }
            memcpy(substr, encoded + pos, try_len);
            substr[try_len] = '\0';
            
            // Search in vocabulary
            int token_id = find_token_index(model, substr);
            free(substr);
            
            if (token_id >= 0) {
                best_token_id = token_id;
                best_token_len = try_len;
                break;  // Found longest match, exit
            }
        }
        
        if (best_token_id >= 0) {
            // Found matching token
            tokens[*n_tokens] = best_token_id;
            (*n_tokens)++;
            pos += best_token_len;
        } else {
            // No match found, skip one character (handle unknown characters)
            // Try to skip one UTF-8 character
            size_t char_len;
            utf8_to_codepoint(encoded + pos, &char_len);
            pos += char_len;
            
            // If there's still space, can add UNK token
            if (*n_tokens < max_tokens) {
                tokens[*n_tokens] = model->vocab.unk_token;
                (*n_tokens)++;
            }
        }
    }
    
    free(encoded);
    return CORTEX_OK;
}

// Get text corresponding to token
// Note: uses static buffer, not thread-safe
const char* cortex_token_to_str(cortex_model *model, cortex_token token) {
    static char buffer[4096];  // Static buffer for storing decoded string
    
    if (!model || token < 0 || token >= (int)model->vocab.size) {
        return NULL;
    }
    
    const char *token_text = model->vocab.tokens[token];
    if (!token_text) {
        return NULL;
    }
    
    // BPE decoding: convert UTF-8 encoded token to original bytes
    char *decoded = decode_bpe_token(token_text);
    if (!decoded) {
        // If decoding fails, return original string
        return token_text;
    }
    
    // Copy to static buffer
    size_t decoded_len = strlen(decoded);
    if (decoded_len >= sizeof(buffer)) {
        decoded_len = sizeof(buffer) - 1;
    }
    memcpy(buffer, decoded, decoded_len);
    buffer[decoded_len] = '\0';
    
    free(decoded);
    return buffer;
}

// Get error message
const char* cortex_error_to_str(cortex_error error) {
    switch (error) {
        case CORTEX_OK:
            return "Success";
        case CORTEX_ERROR_FILE_LOAD:
            return "Failed to load file";
        case CORTEX_ERROR_INVALID_MODEL:
            return "Invalid model format";
        case CORTEX_ERROR_MEMORY:
            return "Memory allocation failed";
        case CORTEX_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case CORTEX_ERROR_TOKENIZATION:
            return "Tokenization failed";
        case CORTEX_ERROR_INFERENCE:
            return "Inference failed";
        default:
            return "Unknown error";
    }
}
