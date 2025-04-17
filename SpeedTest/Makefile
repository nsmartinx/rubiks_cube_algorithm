# —— CPU settings ——
CXX         := g++
CXXFLAGS    := -std=c++20 -O3 -Wall -Wextra
CPU1_TARGET := CpuSpeedTest_1
CPU1_SRC    := CpuSpeedTest_1.cpp
CPU2_TARGET := CpuSpeedTest_2
CPU2_SRC    := CpuSpeedTest_2.cpp

# —— GPU settings ——
NVCC        := nvcc
NVCCFLAGS   := -std=c++20 -O3
GPU1_TARGET := GpuSpeedTest_1
GPU1_SRC    := GpuSpeedTest_1.cu
GPU2_TARGET := GpuSpeedTest_2
GPU2_SRC    := GpuSpeedTest_2.cu

.PHONY: all cpu1 cpu2 gpu1 gpu2 clean

all: cpu1 cpu2 gpu1 gpu2

cpu1: $(CPU1_TARGET)

$(CPU1_TARGET): $(CPU1_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

cpu2: $(CPU2_TARGET)

$(CPU2_TARGET): $(CPU2_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

gpu1: $(GPU1_TARGET)

$(GPU1_TARGET): $(GPU1_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu2: $(GPU2_TARGET)

$(GPU2_TARGET): $(GPU2_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	@rm -f $(CPU1_TARGET) $(CPU2_TARGET) $(GPU1_TARGET) $(GPU2_TARGET)
