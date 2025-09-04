#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "engine_settings.h"

typedef unsigned int uint32;

#define check_cuda_errors(val) check_cuda(val, #val, __FILE__, __LINE__)
/* Вывод отладочной информации работы CUDA, при условии что работа GPU приостанавливается ошибкой. */
void check_cuda(cudaError_t result, const char *const func, const char *const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error: " << static_cast<uint32>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(vec3 *fb, int imgwidth, int imgheight)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if ((i >= imgwidth) || (j >= imgheight)) return;
	
	/* Пример: 1200x600
	   0, 3, 6, ... , 3597
	   3600,    ... , 7197 */
	int pixidx = j*imgwidth + i;
	
	/* Запись градиента */
	fb[pixidx] = vec3(double(i)/imgwidth, double(j)/imgheight, 0.0);
}

int main()
{
	int IMAGE_WIDTH  = 1200,
		IMAGE_HEIGHT = 600;

	int tx = 8,
		ty = 8;

	std::cerr << "Rendering a " << IMAGE_WIDTH << 'x' << IMAGE_HEIGHT << " image ";
	std::cerr << "in " << tx << 'x' << ty << " blocks.\n";

	int num_pix = IMAGE_WIDTH * IMAGE_HEIGHT;
	size_t fb_size = num_pix*sizeof(vec3);

	vec3 *fb;
	check_cuda_errors(cudaMallocManaged((void **)&fb, fb_size));

	clock_t start,
			stop;

	start = clock();

	dim3 blocks(IMAGE_WIDTH/tx + 1, IMAGE_HEIGHT/ty + 1);
	dim3 threads(tx, ty);
	render<<<blocks, threads>>>(fb, IMAGE_WIDTH, IMAGE_HEIGHT);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	stop = clock();

	double timer = ((double)(stop-start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer << " seconds.\n";

	std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
	for (int j = 0; j < IMAGE_HEIGHT; ++j) {
		for (int i = 0; i < IMAGE_WIDTH; ++i) {
			size_t pixidx = j*IMAGE_WIDTH + i;
			write_color(std::cout, &fb[pixidx]);
		}
	}
	check_cuda_errors(cudaFree(fb));
}