TARGETS = mm32 mm32v mm32w mm16tc

all: $(TARGETS)

%: %.cu
	nvcc -O3 $< -o $@ -lcublas

clean:
	rm -rf $(TARGETS)