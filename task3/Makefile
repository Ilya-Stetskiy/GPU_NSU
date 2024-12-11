all:  many_gpu_global.out

many_gpu_global.out:
	nvcc many_gpu_global.cu -o many_gpu_global.out `pkg-config --cflags --libs opencv`

clean:
	rm ./many_gpu_global.out 