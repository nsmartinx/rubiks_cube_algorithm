# —— Directories ——
OUT          := out

# —— Compilers & flags ——
CXX          := g++
CXXFLAGS     := -std=c++20 -O3 -Wall -Wextra

NVCC         := nvcc
NVCCFLAGS    := -std=c++20 -O3

# —— CPU targets ——
CPU1_TARGET  := CpuSpeedTest_1
CPU1_SRC     := CpuSpeedTest_1.cpp
CPU1_BIN     := $(OUT)/$(CPU1_TARGET)

CPU2_TARGET  := CpuSpeedTest_2
CPU2_SRC     := CpuSpeedTest_2.cpp
CPU2_BIN     := $(OUT)/$(CPU2_TARGET)

# —— GPU targets ——
GPU1_TARGET  := GpuSpeedTest_1
GPU1_SRC     := GpuSpeedTest_1.cu
GPU1_BIN     := $(OUT)/$(GPU1_TARGET)

GPU2_TARGET  := GpuSpeedTest_2
GPU2_SRC     := GpuSpeedTest_2.cu
GPU2_BIN     := $(OUT)/$(GPU2_TARGET)

.PHONY: all cpu1 cpu2 gpu1 gpu2 clean

all: cpu1 cpu2 gpu1 gpu2

cpu1: $(CPU1_BIN)

cpu2: $(CPU2_BIN)

gpu1: $(GPU1_BIN)

gpu2: $(GPU2_BIN)

# Build rules
$(CPU1_BIN): $(CPU1_SRC) | $(OUT)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CPU2_BIN): $(CPU2_SRC) | $(OUT)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(GPU1_BIN): $(GPU1_SRC) | $(OUT)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(GPU2_BIN): $(GPU2_SRC) | $(OUT)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Ensure output directory exists
$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)
