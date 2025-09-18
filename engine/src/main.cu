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

__global__ void render_init(vec3* fb, hittable** world, camera** camera)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	(*camera)->render(fb, world, i, j);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int imgwidth, int imgheight)
{
	if (threadIdx.x == 0 &&  blockIdx.x == 0) {
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5);
		*(d_list+1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world	= new hittable_list(d_list,2);

		*d_camera   = new camera();
		(*d_camera)->ASPECT_RATIO = 16.0 / 8.0;
		(*d_camera)->IMAGE_WIDTH  = imgwidth;
		(*d_camera)->IMAGE_HEIGHT = imgheight;
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera)
{
	delete *(d_list);
	delete *(d_list+1);
	delete *d_world;
	delete *d_camera;
}

int main()
{	
	int IMAGE_WIDTH  = 1200,
		IMAGE_HEIGHT = 600;

	int num_pix = IMAGE_WIDTH * IMAGE_HEIGHT;
	size_t fb_size = num_pix*sizeof(vec3);

	vec3* fb;
	check_cuda_errors(cudaMallocManaged((void**)&fb, fb_size));
	
	/* World */
	hittable** d_list;
	check_cuda_errors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*)));
	
	hittable** d_world;
	check_cuda_errors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

	camera** d_camera;
	check_cuda_errors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	create_world<<<1,1>>>(d_list, d_world, d_camera, IMAGE_WIDTH, IMAGE_HEIGHT);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	int tx = 8, ty = 8;

	dim3 blocks(IMAGE_WIDTH/tx + 1, IMAGE_HEIGHT/ty + 1);
	dim3 threads(tx, ty);
	std::cerr << "Rendering a " << IMAGE_WIDTH << 'x' << IMAGE_HEIGHT << " image ";
	std::cerr << "in " << tx << 'x' << ty << " blocks.\n";

	clock_t start, stop;

	start = clock();

	render_init<<<blocks, threads>>>(fb, d_world, d_camera);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	stop = clock();

	/* Print in .ppm file */
	double timer = ((double)(stop-start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer << " seconds.\n";

	std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
	for (int j = 0; j < IMAGE_HEIGHT; ++j) {
		for (int i = 0; i < IMAGE_WIDTH; ++i) {
			size_t pixidx = j*IMAGE_WIDTH + i;
			write_color(std::cout, &fb[pixidx]);
		}
	}

	free_world<<<1,1>>>(d_list, d_world, d_camera);
	check_cuda_errors(cudaDeviceSynchronize());
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaFree(d_list));
	check_cuda_errors(cudaFree(d_world));
	check_cuda_errors(cudaFree(d_camera));
	check_cuda_errors(cudaFree(fb));
	cudaDeviceReset();
}