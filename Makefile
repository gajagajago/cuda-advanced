TARGETS = mm32 mm32v mm32w mm32wv mm16 mm16tc

all: $(TARGETS)

%: %.cu
	nvcc -O3 $< -o $@ -lcublas -arch=sm_70

clean:
	rm -rf $(TARGETS)