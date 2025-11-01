#include "chat_template.h"

#include "../pkg/minja/chat-template.hpp"
#include "../pkg/nlohmann/json.hpp"
#include <string>
#include <memory>
#include <cstring>
#include <cstdio>

using json = nlohmann::ordered_json;

// Internal structure to hold chat template object
struct chat_template_impl {
    std::unique_ptr<minja::chat_template> template_obj;
    
    chat_template_impl(const std::string& source, const std::string& bos_token, const std::string& eos_token)
        : template_obj(std::make_unique<minja::chat_template>(source, bos_token, eos_token))
    {
    }
};

extern "C" {

chat_template_handle chat_template_load_from_string(
    const char* template_str,
    const char* bos_token,
    const char* eos_token)
{
    if (!template_str) {
        return nullptr;
    }
    
    try {
        std::string bos = bos_token ? std::string(bos_token) : "";
        std::string eos = eos_token ? std::string(eos_token) : "";
        
        auto impl = new chat_template_impl(std::string(template_str), bos, eos);
        return reinterpret_cast<chat_template_handle>(impl);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to load chat template: %s\n", e.what());
        return nullptr;
    }
}

char* chat_template_render(
    chat_template_handle handle,
    const char* messages_json,
    bool add_generation_prompt,
    char* output_buffer,
    size_t buffer_size)
{
    if (!handle || !messages_json || !output_buffer || buffer_size == 0) {
        return nullptr;
    }
    
    try {
        auto impl = reinterpret_cast<chat_template_impl*>(handle);
        
        // Parse messages JSON
        json messages = json::parse(messages_json);
        if (!messages.is_array()) {
            fprintf(stderr, "messages_json must be a JSON array\n");
            return nullptr;
        }
        
        // Prepare inputs
        minja::chat_template_inputs inputs;
        inputs.messages = messages;
        inputs.add_generation_prompt = add_generation_prompt;
        
        // Render template
        minja::chat_template_options opts;
        opts.apply_polyfills = true;
        std::string result = impl->template_obj->apply(inputs, opts);
        
        // Copy to output buffer
        if (result.length() >= buffer_size) {
            fprintf(stderr, "Rendered template too long (%zu >= %zu)\n", result.length(), buffer_size);
            return nullptr;
        }
        
        strncpy(output_buffer, result.c_str(), buffer_size - 1);
        output_buffer[buffer_size - 1] = '\0';
        
        return output_buffer;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to render chat template: %s\n", e.what());
        return nullptr;
    }
}

void chat_template_free(chat_template_handle handle)
{
    if (handle) {
        delete reinterpret_cast<chat_template_impl*>(handle);
    }
}

} // extern "C"

