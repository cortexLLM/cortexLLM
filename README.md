# cortexLLM

A lightweight C/C++ LLM inference engine specialized for **Qwen3** models in GGUF format.

## Features

- 🚀 **High Performance**: Optimized C++ implementation with multi-threaded inference using OpenMP
- 📦 **Lightweight**: Minimal dependencies, built on GGML backend
- 🔧 **GGUF Support**: Full support for GGUF format Qwen3 model files
- 🧠 **Quantization**: Optimized support for **Q8_0** quantization format
- 💬 **Interactive**: Supports both chat and generate modes
- 🎯 **Qwen3 Optimized**: Specifically designed for Qwen3 architecture with QK-RMSNorm and RoPE
- 📝 **Chat Templates**: Built-in support for Jinja2-style chat templates
- 🔄 **Flexible Prompting**: Supports system prompts and custom chat templates

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Command Line Options](#command-line-options)
- [Supported Models](#supported-models)
- [Build Instructions](#build-instructions)
- [Usage Examples](#usage-examples)
- [Architecture Details](#architecture-details)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## System Requirements

- Linux/macOS/Windows (WSL)
- CMake 3.14+
- C++17 compatible compiler (GCC 7.0+, Clang 5.0+, or MSVC 2017+)
- At least 4GB RAM (8GB+ recommended for larger models)
- CPU with AVX2 support (recommended for better performance)

## Quick Start

### 1. Build from Source

```bash
# Clone the repository
git clone https://github.com/cortexLLM/cortexLLM.git
cd cortexLLM

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
cmake --build . -j$(nproc)

# The executable will be in build/bin/cortexllm
```

### 2. Basic Usage

```bash
# Interactive chat mode (default)
./build/bin/cortexllm model.gguf

# Generate mode with prompt
./build/bin/cortexllm model.gguf -m generate -i "Hello, how are you?"

# Chat mode with system prompt
./build/bin/cortexllm model.gguf -m chat -i "Tell me a joke" -y "You are a helpful assistant"
```

## Command Line Options

```
Usage: cortexllm <checkpoint> [options]

Options:
  -t <float>  Temperature for sampling [0,inf], default: 1.0
              (0 = greedy/deterministic, 1.0 = default, higher = more creative)
  -p <float>  Top-p (nucleus) sampling threshold [0,1], default: 0.9
              (1.0 = disabled, 0.9 = recommended)
  -s <int>    Random seed for generation, default: current time
  -c <int>    Context window size, default: 0 (uses model's max sequence length)
  -m <string> Mode: "generate" or "chat", default: "chat"
  -i <string> Input prompt text
  -y <string> System prompt (used in chat mode)
  -r <int>    Reasoning mode: 0 = disabled (default), 1 = enabled
  -T <string> Override chat template from GGUF with custom Jinja2 template
  -j <int>    Jinja template mode: -1 = auto (default), 0 = disable, 1 = enable
```

### Option Details

- **Temperature (-t)**: Controls randomness. Lower values make output more deterministic.
- **Top-p (-p)**: Nucleus sampling - samples from tokens with cumulative probability ≥ p.
- **Context Size (-c)**: Limit context window. Set to 0 to use model's maximum.
- **Mode (-m)**: 
  - `generate`: Simple text generation from prompt
  - `chat`: Interactive conversation with turn-taking
- **System Prompt (-y)**: Optional system message for chat mode.
- **Custom Template (-T)**: Override model's chat template with custom Jinja2 template string.

## Supported Models

- ✅ **Qwen3 Series**: Fully supported and optimized
  - Qwen3-0.6B, Qwen3-1.5B, Qwen3-2B, etc.
  - Supports both base and instruction-tuned variants

**Note**: This implementation is specifically optimized for Qwen3 architecture. Other models may not work correctly.

## Quantization Formats

### Currently Supported

- ✅ **Q8_0** (8-bit quantization)
  - 32 elements per block with FP16 scale factor
  - Fully optimized inference path
  - Primary and recommended format

### Planned Support (Under Development)

- ⏳ **F32** (32-bit floating point)
- ⏳ **F16** (16-bit floating point)
- ⏳ **Q4_0** (4-bit quantization)
- ⏳ **Q4_1** (4-bit quantization)
- ⏳ **Q5_0** (5-bit quantization)
- ⏳ **Q5_1** (5-bit quantization)

**Note**: While GGML supports many quantization formats, cortexLLM's custom inference engine currently implements only Q8_0. The inference path is specifically optimized for Q8_0 quantization, providing the best performance for this format. Other formats will be added in future releases.

## Build Options

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

### Debug Build

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j$(nproc)
```

### Release Build (Optimized)

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

### Build Options

# Memory check
make memcheck
```

## Usage Examples

### Basic Chat

```bash
$ ./cortexllm Qwen3-0.6B-Q8_0.gguf -m chat -i "What is machine learning?"

> What is machine learning?
Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed...
```

### Generation Mode

```bash
$ ./cortexllm Qwen3-0.6B-Q8_0.gguf -m generate -i "The future of AI" -t 0.7 -p 0.95

The future of AI holds tremendous potential for transforming industries...
```

### With System Prompt

```bash
$ ./cortexllm model.gguf -m chat -i "Write a poem" -y "You are a creative poet"
```

### Custom Temperature and Top-p

```bash
$ ./cortexllm model.gguf -i "Tell a story" -t 0.8 -p 0.9
```

### Limited Context Window

```bash
$ ./cortexllm model.gguf -i "Long prompt..." -c 512
```

### Custom Chat Template

```bash
$ ./cortexllm model.gguf -T "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
```

## Architecture Details

cortexLLM implements a custom inference engine optimized for Qwen3 models:

### Qwen3-Specific Features

- **QK-RMSNorm**: Separate RMSNorm for Query and Key heads
- **RoPE**: Rotary Position Embedding with configurable base frequency
- **Grouped-Query Attention**: Support for fewer KV heads than query heads
- **SwiGLU**: Swish-Gated Linear Unit activation in FFN layers

### Quantization

- Uses group-wise quantization (32 elements per group for Q8_0)
- Quantized matrix-vector multiplication with per-group scaling
- Efficient memory layout for cache-friendly access

### Performance Optimizations

- OpenMP parallelization for attention and matrix operations
- KV cache for efficient autoregressive generation
- Optimized memory allocation patterns

## Troubleshooting

### Common Issues

1. **Model Loading Failure**
   ```
   Failed to load GGUF file
   ```
   - Ensure the model file is a valid GGUF format Qwen3 model
   - Check file path is correct
   - Verify sufficient memory available

2. **Unsupported Architecture Error**
   ```
   Unsupported architecture: xxx (supported: 'qwen3')
   ```
   - This implementation only supports Qwen3 models
   - Convert your model or use a different inference engine

3. **Compilation Errors**
   - Ensure CMake 3.14+ is installed
   - Check C++17 compiler support: `g++ --version` or `clang++ --version`
   - Install build dependencies: `sudo apt install build-essential cmake` (Linux)

4. **Performance Issues**
   - Ensure CPU supports AVX2: `grep avx2 /proc/cpuinfo`
   - Use appropriate number of threads (OpenMP auto-detects by default)
   - Consider using quantized models (Q8_0) for better memory efficiency

5. **Template Rendering Errors**
   ```
   Warning: Template render failed, falling back to sprintf
   ```
   - Check Jinja2 template syntax if using custom templates
   - Ensure BOS/EOS tokens are correctly configured

### Debug Mode

```bash
# Build with debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j$(nproc)

# Run with verbose output
./cortexllm model.gguf -i "test" 2>&1 | tee debug.log
```

### Memory Check

```bash
# Build with sanitizers
cmake -DCORTEX_SANITIZE_ADDRESS=ON -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j$(nproc)
```

## Development

### Project Structure

```
cortexLLM/
├── src/
│   ├── main.cpp              # Main entry point and inference loop
│   ├── gguf-loader.cpp       # GGUF file loading utilities
│   └── gguf_token.cpp        # Tokenizer implementation
├── include/
│   └── gguf-loader.h         # GGUF loader header
├── utils/
│   ├── log.h / log.cpp       # Logging utilities
│   └── chat_template.h / .cpp # Chat template rendering
├── pkg/
│   └── ggml/                 # GGML backend (submodule)
├── CMakeLists.txt            # Main build configuration
└── README.md                 # This file
```

### Key Components

- **main.cpp**: Complete inference engine with forward pass, sampling, and generation loops
- **GGUF Loader**: Reads Qwen3 model architecture and weights from GGUF files
- **Tokenizer**: BPE tokenization with special token handling
- **Chat Templates**: Jinja2-style template rendering for conversation formatting
- **Sampler**: Supports greedy, multinomial, and top-p (nucleus) sampling

### Code Style

- C++17 standard
- Extensive English comments (recently updated from Chinese)
- Doxygen-style documentation comments

### Contributing

Issues and Pull Requests are welcome!

When contributing:
1. Follow existing code style
2. Add English comments for new functions
3. Update this README if adding features
4. Test with Qwen3 models

## License

See LICENSE file for details.
