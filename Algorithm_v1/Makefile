# —— Directories ——
OUT          := out

# —— Compilers & flags ——
CXX          := g++
CXXFLAGS     := -std=c++20 -O3 -Wall -Wextra

NVCC         := nvcc
NVCCFLAGS    := -std=c++20 -O3 -Xcompiler="-Wall -Wextra"

# —— Target ——
TARGET       := rubiks
BIN          := $(OUT)/$(TARGET)

# —— Sources ——
SRC_CPP      := main.cpp
SRC_CU       := stage1.cu stage2.cu stage3.cu stage4.cu

# —— Objects ——
OBJ_CPP      := $(OUT)/main.o
OBJ_CU       := $(patsubst %.cu,$(OUT)/%.o,$(SRC_CU))
OBJS         := $(OBJ_CPP) $(OBJ_CU)

.PHONY: all clean

# Default target
all: $(BIN)

# Link everything into the final executable
$(BIN): $(OBJS) | $(OUT)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

# Compile main.cpp
$(OUT)/main.o: main.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile each CUDA source into an object
$(OUT)/%.o: %.cu | $(OUT)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Ensure output directory exists
$(OUT):
	mkdir -p $(OUT)

# Clean up
clean:
	rm -rf $(OUT)
