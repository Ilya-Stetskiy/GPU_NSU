all: global.out shared.out texture.out 

global.out:
	nvcc global.cu -o global.out `pkg-config --cflags --libs opencv`

shared.out:
	nvcc shared.cu -o shared.out `pkg-config --cflags --libs opencv`

texture.out:
	nvcc texture.cu -o texture.out `pkg-config --cflags --libs opencv`

clean:
	rm ./global.out ./shared.out ./texture.out 