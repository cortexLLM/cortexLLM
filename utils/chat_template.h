#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for chat template
typedef void* chat_template_handle;

// Load chat template from string
// template_str: Jinja2 template string
// bos_token: Beginning of sequence token string (can be NULL)
// eos_token: End of sequence token string (can be NULL)
// Returns: handle on success, NULL on failure
chat_template_handle chat_template_load_from_string(
    const char* template_str,
    const char* bos_token,
    const char* eos_token
);

// Render chat template with messages
// handle: Template handle from chat_template_load_from_string
// messages_json: JSON array of messages, e.g. [{"role": "user", "content": "..."}]
// add_generation_prompt: Whether to add generation prompt
// output_buffer: Buffer to store rendered result
// buffer_size: Size of output buffer
// Returns: pointer to output_buffer on success, NULL on failure
char* chat_template_render(
    chat_template_handle handle,
    const char* messages_json,
    bool add_generation_prompt,
    char* output_buffer,
    size_t buffer_size
);

// Free chat template handle
void chat_template_free(chat_template_handle handle);

#ifdef __cplusplus
}
#endif

