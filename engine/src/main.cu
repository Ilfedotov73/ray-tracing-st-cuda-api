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

__device__ color ray_color(const ray& r)
{
	vec3 unit_direction = unitv(r.direction());
	double a = 0.5*(unit_direction.y() + 1.0);
	return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int imgwidth, int imgheight, vec3 pixel00_loc, 
					   vec3 pixel_delta_u, vec3 pixel_delta_v, point3 camera_center)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	if ((i >= imgwidth) || (j >= imgheight)) return;
	
	int pixidx = j*imgwidth + i;
	point3 pixel = pixel00_loc + (i*pixel_delta_u) + (j*pixel_delta_v);
	vec3 ray_direction = pixel - camera_center;

	ray r(camera_center, ray_direction);
	fb[pixidx] = ray_color(r);
}

int main()
{
	/* Image */
	double ASPECT_RATIO = 16.9 / 9.0;
	int IMAGE_WIDTH  = 1200;

	int IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO);
	IMAGE_HEIGHT = (IMAGE_HEIGHT < 1) ? 1 : IMAGE_HEIGHT;

	/* Camera */
	double FOCAL_LENGTH = 1.0;
	double VIEWPORT_HEIGHT = 2.0;
	double VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (double(IMAGE_WIDTH)/IMAGE_HEIGHT);
	point3 CAMERA_CENTER = point3(0,0,0);

	vec3 VIEWPORT_U = vec3(VIEWPORT_WIDTH, 0, 0);
	vec3 VIEWPORT_V = vec3(0, -VIEWPORT_HEIGHT, 0);

	vec3 PIXEL_DELTA_U = VIEWPORT_U / IMAGE_WIDTH;
	vec3 PIXEL_DELTA_V = VIEWPORT_V / IMAGE_HEIGHT;

	point3 VIEWPORT_UPPER_LEFT = CAMERA_CENTER - vec3(0, 0, FOCAL_LENGTH) - VIEWPORT_U/2 - VIEWPORT_V/2;
	point3 PIXEL00_LOC = VIEWPORT_UPPER_LEFT + 0.5 * (PIXEL_DELTA_U + PIXEL_DELTA_V);

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
	render<<<blocks, threads>>>(fb, IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL00_LOC, PIXEL_DELTA_U,
								PIXEL_DELTA_V, CAMERA_CENTER);
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