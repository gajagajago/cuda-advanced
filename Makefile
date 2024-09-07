all: mm32 mm32v mm32w mm16tc

mm32: mm32.cu
	nvcc -O3 $< -o $@ -lcublas

mm32v: mm32v.cu
	nvcc -O3 $< -o $@ -lcublas

mm32w: mm32w.cu
	nvcc -O3 $< -o $@ -lcublas

mm16tc: mm16tc.cu
	nvcc -O3 $< -o $@ -lcublas

clean:
	rm -rf mm32 mm32v mm32w mm16tc
