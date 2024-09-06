all: mm32 mm32v mm16

mm32: mm32.cu
	nvcc -O3 $< -o $@ -lcublas

mm32v: mm32v.cu
	nvcc -O3 $< -o $@ -lcublas

mm16: mm16.cu
	nvcc -O3 $< -o $@ -lcublas

clean:
	rm -rf mm32 mm32v mm16
