TARGETS = mm32 mm32v mm32cv mm32w mm32wv mm16tc

all: $(TARGETS)

%: %.cu
	nvcc -O3 $< -o $@ -lcublas -arch=sm_70

clean:
	rm -rf $(TARGETS)