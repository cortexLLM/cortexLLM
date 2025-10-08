#include "cortex_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Print usage information
void print_usage(const char *program_name) {
    printf("CortexLLM - A lightweight LLM inference engine\n");
    printf("Usage: %s <model_file> [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  -n, --n-predict <count>   Set number of tokens to generate (default: 100)\n");
    printf("  -p, --prompt <text>       Set input prompt\n");
    printf("  -h, --help               Show this help information\n");
    printf("\nExamples:\n");
    printf("  %s model.gguf -p \"Hello, world!\"\n", program_name);
    printf("  %s model.gguf -n 200\n", program_name);
}

// Parse command line arguments
int parse_args(int argc, char *argv[], char **model_path, 
               int *n_predict, char **prompt) {
    *model_path = NULL;
    *n_predict = 100;
    *prompt = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n-predict") == 0) {
            if (i + 1 < argc) {
                *n_predict = atoi(argv[++i]);
                if (*n_predict <= 0) {
                    fprintf(stderr, "Error: generation count must be greater than 0\n");
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: -n option requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) {
                *prompt = argv[++i];
            } else {
                fprintf(stderr, "Error: -p option requires an argument\n");
                return -1;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: unknown option %s\n", argv[i]);
            return -1;
        } else {
            if (*model_path == NULL) {
                *model_path = argv[i];
            } else {
                fprintf(stderr, "Error: only one model file can be specified\n");
                return -1;
            }
        }
    }
    
    if (*model_path == NULL) {
        fprintf(stderr, "Error: model file must be specified\n");
        print_usage(argv[0]);
        return -1;
    }
    
    return 1;
}

// Generate text
void generate_text(cortex_context *ctx, const char *prompt, int n_predict) {
    printf("Prompt: %s\n", prompt);
    printf("Generated: ");
    
    // Tokenization
    cortex_token tokens[1024];
    int n_tokens;
    cortex_error err = cortex_tokenize(ctx->model, prompt, tokens, 1024, &n_tokens);
    if (err != CORTEX_OK) {
        fprintf(stderr, "Tokenization failed: %s\n", cortex_error_to_str(err));
        return;
    }
    
    printf("Tokenization result: %d tokens:", n_tokens);
    for (int i = 0; i < n_tokens && i < 10; i++) {
        printf(" %d", tokens[i]);
    }
    printf("\n");
    
    // Initial inference
    err = cortex_eval(ctx, tokens, n_tokens, 0);
    if (err != CORTEX_OK) {
        fprintf(stderr, "Inference failed: %s\n", cortex_error_to_str(err));
        return;
    }
    
    
    // Generation loop
    cortex_pos n_past = n_tokens;
    for (int i = 0; i < n_predict; i++) {
        // Diagnostic: print first generation logits
        if (i == 0) {
            printf("\n[CORTEX] Key logits:\n");
            printf("  Token 15846: %.4f\n", ctx->logits[15846]);
            printf("  Token 3988: %.4f\n", ctx->logits[3988]);
            printf("  Token 21806: %.4f\n", ctx->logits[21806]);
            printf("  Token 38297: %.4f\n", ctx->logits[38297]);
            printf("  Token 7662: %.4f\n", ctx->logits[7662]);
            
            // Find top-5
            struct {int idx; float val;} top5[5] = {{0, -1e9f}};
            for (int k = 0; k < ctx->model->hparams.n_vocab; k++) {
                for (int j = 0; j < 5; j++) {
                    if (ctx->logits[k] > top5[j].val) {
                        for (int m = 4; m > j; m--) top5[m] = top5[m-1];
                        top5[j].idx = k;
                        top5[j].val = ctx->logits[k];
                        break;
                    }
                }
            }
            
            printf("  Top-5:\n");
            for (int j = 0; j < 5; j++) {
                const char *token_str = cortex_token_to_str(ctx->model, top5[j].idx);
                printf("    #%d: token=%d, logit=%.4f, text='%s'\n", j, top5[j].idx, top5[j].val, token_str ? token_str : "NULL");
            }
            printf("\n");
        }
        
        // Sample next token (greedy: temp=0, no top-k)
        cortex_token next_token = cortex_sample(ctx, 0.0f, 0.9f, 0);
        
        // Diagnostics disabled
        
        // Diagnostics disabled
        
        // Output token
        const char *token_str = cortex_token_to_str(ctx->model, next_token);
        if (token_str) {
            printf("%s", token_str);
            fflush(stdout);
        } else {
            printf("[UNK]");
        }
        
        // Diagnostics disabled
        
        // Diagnostics disabled
        
        // Check termination condition
        if (next_token == ctx->model->vocab.eos_token) {
            printf("\n[EOS]\n");
            break;
        }
        
        // Continue inference
        err = cortex_eval(ctx, &next_token, 1, n_past);
        if (err != CORTEX_OK) {
            fprintf(stderr, "Inference failed: %s\n", cortex_error_to_str(err));
            break;
        }
        
        n_past++;
    }
    
    printf("\n");
}

// Interactive mode
void interactive_mode(cortex_context *ctx) {
    char input[1024];
    printf("\nEntering interactive mode (type 'quit' to exit):\n");
    
    while (1) {
        printf("> ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        // Remove newline character
        input[strcspn(input, "\n")] = '\0';
        
        if (strcmp(input, "quit") == 0) {
            break;
        }
        
        if (strlen(input) > 0) {
            generate_text(ctx, input, 50);
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    char *model_path;
    int n_predict;
    char *prompt;
    
    int parse_result = parse_args(argc, argv, &model_path, &n_predict, &prompt);
    if (parse_result <= 0) {
        return parse_result == 0 ? 0 : 1;
    }
    
    // Load model
    cortex_model *model;
    cortex_error err = cortex_model_load(model_path, &model);
    if (err != CORTEX_OK) {
        fprintf(stderr, "Model loading failed: %s\n", cortex_error_to_str(err));
        return 1;
    }
    
    // Create context
    cortex_context *ctx;
    err = cortex_context_new(model, &ctx);
    if (err != CORTEX_OK) {
        fprintf(stderr, "Context creation failed: %s\n", cortex_error_to_str(err));
        cortex_model_free(model);
        return 1;
    }
    
    // Execute inference
    if (prompt) {
        generate_text(ctx, prompt, n_predict);
    } else {
        interactive_mode(ctx);
    }
    
    // Clean up resources
    cortex_context_free(ctx);
    cortex_model_free(model);
    
    return 0;
}
