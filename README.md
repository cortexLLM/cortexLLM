# cortexLLM

A lightweight C language LLM inference engine supporting GGUF format quantized models.

## Features

- 🚀 **High Performance**: Optimized C implementation with multi-threaded inference
- 📦 **Lightweight**: Minimal dependencies for easy deployment
- 🔧 **GGUF Support**: Full support for GGUF format model files
- 🧠 **Quantization Support**: Supports Q8_0, Q4_0, and other quantization formats
- 💬 **Interactive**: Supports interactive conversation mode
- 🎯 **Inference Focused**: Specifically optimized for inference, no training code included

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Command Line Options](#command-line-options)
- [Supported Models](#supported-models)
- [Supported Quantization Formats](#supported-quantization-formats)
- [Build Options](#build-options)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)

## System Requirements

- Linux/macOS/Windows (WSL)
- GCC 7.0+ or Clang 5.0+
- At least 4GB RAM (8GB+ recommended)
- CPU with AVX2 support (recommended)

## Quick Start

### 1. Compilation

```bash
# Clone the project
cd cortexLLM

# Compile
make

# Or compile release version
make release
```

### 2. Running

```bash
# Basic usage
./cortexLLM model.gguf

# Specify prompt text
./cortexLLM model.gguf -p "Hello, world!"

# Set number of tokens to generate
./cortexLLM model.gguf -p "Please introduce artificial intelligence" -n 200

# Set number of threads
./cortexLLM model.gguf -t 8 -n 100
```

### 3. Interactive Mode

```bash
# Entering interactive mode when no prompt text is specified
./cortexLLM model.gguf
```

## Command Line Options

```
Usage: cortexLLM <model file> [options]

Options:
  -t, --threads <number>     Set number of threads (default: 4)
  -n, --n-predict <number>   Set number of tokens to generate (default: 100)
  -p, --prompt <text>        Set input prompt
  -h, --help                 Show help information
```

## Supported Models

- ✅ Qwen2/Qwen2.5 Series
- ✅ LLaMA Series
- ✅ Mistral Series
- ✅ Other Transformer-based Models

## Supported Quantization Formats

- ✅ F32 (32-bit floating point)
- ✅ F16 (16-bit floating point)
- ✅ Q8_0 (8-bit quantization)
- ✅ Q4_0 (4-bit quantization)
- ✅ Q4_1 (4-bit quantization)
- ✅ Q5_0 (5-bit quantization)
- ✅ Q5_1 (5-bit quantization)

## Build Options

```bash
# Debug version
make debug

# Release version (optimized)
make release

# Clean build files
make clean

# Install to system
make install

# Run tests
make test

# Performance benchmark
make benchmark

# Memory check
make memcheck
```

## Performance Optimization

### Compilation Optimization

```bash
# Optimize for specific CPU
make CFLAGS="-O3 -march=native -mtune=native"

# Enable all optimizations
make release
```

### Runtime Optimization

- Use appropriate number of threads (typically CPU core count)
- Ensure sufficient memory is available
- Store model files on SSD

## Examples

### Basic Conversation

```bash
$ ./cortexLLM qwen2-0.5b-instruct.gguf -p "Please introduce deep learning"
```

### Code Generation

```bash
$ ./cortexLLM codellama-7b.gguf -p "Write a C implementation of quicksort"
```

### Interactive Usage

```bash
$ ./cortexLLM model.gguf
Entering interactive mode. Type 'quit' or 'exit' to exit.
Type 'clear' to clear context.

User: Hello
Assistant: Hello! How can I help you today?

User: Please explain what machine learning is
Assistant: Machine learning is a branch of artificial intelligence...
```

## Troubleshooting

### Common Issues

1. **Model loading failure**
   - Check if the model file path is correct
   - Ensure the model file is in valid GGUF format
   - Check if there is sufficient memory

2. **Compilation errors**
   - Ensure GCC version >= 7.0
   - Install necessary development tools: `sudo apt install build-essential`

3. **Performance issues**
   - Adjust thread count: `-t 4`
   - Use quantized models to reduce memory usage
   - Ensure CPU supports AVX2 instruction set

### Debug Mode

```bash
# Compile debug version
make debug

# Run memory check
make memcheck

# View detailed output
./cortexLLM model.gguf -p "test" 2>&1 | tee debug.log
```

## Development

### Project Structure

```
cortexLLM/
├── include/
│   └── cortex_llm.h      # Main header file
├── src/
│   ├── main.c            # Main program
│   ├── model_loader.c    # Model loader
│   ├── inference.c       # Inference engine
│   ├── tokenizer.c       # Tokenizer
│   └── gguf_loader.c     # GGUF file loader
├── Makefile              # Build file
└── README.md             # Documentation
```

## Contributing

Issues and Pull Requests are welcome!

