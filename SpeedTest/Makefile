# —— CPU settings ——
CXX        := g++
CXXFLAGS   := -std=c++20 -O3 -Wall -Wextra
CPU1_TARGET := CpuSpeedTest_1
CPU1_SRC    := CpuSpeedTest_1.cpp

# —— GPU settings ——
NVCC        := nvcc
NVCCFLAGS   := -std=c++20 -O3
GPU_TARGET  := GpuSpeedTest
GPU_SRC     := GpuSpeedTest.cu

.PHONY: all cpu1 gpu clean

all: cpu gpu

cpu1: $(CPU1_TARGET)

$(CPU1_TARGET): $(CPU1_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

gpu: $(GPU_TARGET)

$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	@rm -f $(CPU_TARGET) $(GPU_TARGET)
