# CortexLLM Makefile
# A lightweight LLM inference engine

# choose your compiler, e.g. gcc/clang
# example override to clang: make CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: all
all: cortexLLM

# source files
SOURCES = src/main.c src/model_loader.c src/inference.c src/tokenizer.c src/gguf_loader.c

# build the main executable
cortexLLM: $(SOURCES)
	$(CC) -O3 -std=c99 -Wall -Wextra -march=native -mtune=native -fPIC -fno-strict-aliasing -D_GNU_SOURCE -D_XOPEN_SOURCE=700 -Iinclude -I. -o cortexLLM $(SOURCES) -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./cortexLLM test_model.gguf -p "Hello" -n 3
.PHONY: debug
debug: $(SOURCES)
	$(CC) -g -std=c99 -Wall -Wextra -O0 -DDEBUG -Iinclude -I. -o cortexLLM $(SOURCES) -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: fast
fast: $(SOURCES)
	$(CC) -Ofast -std=c99 -Wall -Wextra -march=native -mtune=native -fPIC -fno-strict-aliasing -D_GNU_SOURCE -D_XOPEN_SOURCE=700 -Iinclude -I. -o cortexLLM $(SOURCES) -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./cortexLLM test_model.gguf
.PHONY: openmp
openmp: $(SOURCES)
	$(CC) -Ofast -fopenmp -std=c99 -Wall -Wextra -march=native -mtune=native -fPIC -fno-strict-aliasing -D_GNU_SOURCE -D_XOPEN_SOURCE=700 -Iinclude -I. -o cortexLLM $(SOURCES) -lm

.PHONY: clean
clean:
	rm -f cortexLLM



# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: gnu
gnu: $(SOURCES)
	$(CC) -Ofast -std=gnu11 -Wall -Wextra -march=native -mtune=native -fPIC -fno-strict-aliasing -D_GNU_SOURCE -D_XOPEN_SOURCE=700 -Iinclude -I. -o cortexLLM $(SOURCES) -lm

.PHONY: gnuopenmp
gnuopenmp: $(SOURCES)
	$(CC) -Ofast -fopenmp -std=gnu11 -Wall -Wextra -march=native -mtune=native -fPIC -fno-strict-aliasing -D_GNU_SOURCE -D_XOPEN_SOURCE=700 -Iinclude -I. -o cortexLLM $(SOURCES) -lm

# run tests if test model is available
.PHONY: test
test: cortexLLM
	@echo "Running tests..."
	@if [ -f "test_model.gguf" ]; then \
		./cortexLLM test_model.gguf -p "Hello, world!" -n 10; \
	else \
		echo "Warning: test model file test_model.gguf not found"; \
		echo "Please place your model file in current directory and rename it to test_model.gguf"; \
	fi

# run performance benchmark
.PHONY: benchmark
benchmark: cortexLLM
	@echo "Running performance benchmark..."
	@if [ -f "test_model.gguf" ]; then \
		time ./cortexLLM test_model.gguf -p "The quick brown fox" -n 50 -t 1; \
		time ./cortexLLM test_model.gguf -p "The quick brown fox" -n 50 -t 4; \
	else \
		echo "Warning: test model file test_model.gguf not found"; \
	fi

# run memory check with valgrind
.PHONY: memcheck
memcheck: debug
	@echo "Running memory check..."
	@if [ -f "test_model.gguf" ]; then \
		valgrind --leak-check=full --show-leak-kinds=all ./cortexLLM test_model.gguf -p "Test" -n 5; \
	else \
		echo "Warning: test model file test_model.gguf not found"; \
	fi

# run static analysis
.PHONY: analyze
analyze:
	@echo "Running static analysis..."
	cppcheck --enable=all --inconclusive --std=c99 src/ include/
	@echo "Static analysis completed"

# format code
.PHONY: format
format:
	@echo "Formatting code..."
	clang-format -i src/*.c include/*.h
	@echo "Code formatting completed"

# show help information
.PHONY: help
help:
	@echo "CortexLLM Build System"
	@echo "======================"
	@echo "Available targets:"
	@echo "  all        - Build project (default)"
	@echo "  clean      - Clean build files"
	@echo "  debug      - Build debug version"
	@echo "  fast       - Build with -Ofast optimization"
	@echo "  openmp     - Build with OpenMP support"
	@echo "  gnu        - Build with gnu11 standard"
	@echo "  gnuopenmp  - Build with gnu11 standard and OpenMP"
	@echo "  test       - Run tests"
	@echo "  benchmark  - Run performance benchmark"
	@echo "  memcheck   - Run memory check with valgrind"
	@echo "  analyze    - Run static analysis"
	@echo "  format     - Format code"
	@echo "  help       - Show this help information"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build project"
	@echo "  make debug             # Build debug version"
	@echo "  make test              # Run tests"
	@echo "  make clean && make     # Rebuild"
