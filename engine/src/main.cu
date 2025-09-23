#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
__device__ vec3 random_unit_vector(curandState* local_rand_state)
{
	for (;;) {
		vec3 p = 2.0 * RANDVEC3 - vec3(1,1,1);
		double lensq = p.length();
		if (1e-160 < lensq && lensq <= 1) { return p / sqrt(lensq); }
	}
}

__device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state)
{
	ray current_ray = r;
	double attenuation = 1.0;
	for (int i = 0; i < 50; ++i) {
		hit_record rec;
		if ((*world)->hit(current_ray, 0.001, DBL_MAX, rec)) {
			vec3 target = rec.p + rec.normal + random_unit_vector(local_rand_state);
			attenuation *= 0.5;
			current_ray = ray(rec.p, target-rec.p);
		}
		else {
			vec3 unit_direction = unitv(current_ray.direction());
			double a = 0.5 * (unit_direction.y() + 1.0);
			vec3 c = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
			return attenuation * c;
		}
	}
	return color(0.0,0.0,0.0);
}

__global__ void rnd_render_init(int imgwidth, int imgheight, curandState* rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= imgwidth) || (j >= imgheight)) { return; }
	int pixidx = j*imgwidth+i;
	curand_init(1984, pixidx, 0, &rand_state[pixidx]);
}

__global__ void render(vec3* fb, hittable** world, camera** cam, curandState* rand_state, int imgwidth, int imgheight,
					   int samples_per_pixel)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= imgwidth) || (j >= imgheight)) { return; }
	int pixidx = j*imgwidth+i;

	curandState local_rand_state = rand_state[pixidx];

	color pixel_color(0,0,0);
	for (int sample = 0; sample < samples_per_pixel; ++sample) {
		double u = (i + curand_uniform_double(&local_rand_state) - 0.5) / imgwidth;
		double v = (j + curand_uniform_double(&local_rand_state) - 0.5) / imgheight;
		ray r = (*cam)->get_ray(u,v);
		pixel_color += ray_color(r, world, &local_rand_state);
	}
	fb[pixidx] = pixel_color / samples_per_pixel;
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera)
{
	if (threadIdx.x == 0 &&  blockIdx.x == 0) {
		*(d_list)   = new sphere(vec3(0,0,-1), 0.5);
		*(d_list+1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world	= new hittable_list(d_list,2);
		*d_camera   = new camera();
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
	size_t new_stack_size = 2 * 1024;
	check_cuda_errors(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));
	std::cerr << "Stack size limit: " << new_stack_size << " bytes" << "\n";

	int IMAGE_WIDTH       = 1200,
		IMAGE_HEIGHT      = 600,
		SAMPLES_PER_PIXEL = 100;

	int num_pix = IMAGE_WIDTH * IMAGE_HEIGHT;
	size_t fb_size = num_pix*sizeof(vec3);

	vec3* fb;
	check_cuda_errors(cudaMallocManaged((void**)&fb, fb_size));
	
	curandState* d_rand_state;
	check_cuda_errors(cudaMalloc((void**)&d_rand_state, num_pix*sizeof(curandState)));

	/* World */
	hittable** d_list;
	check_cuda_errors(cudaMalloc((void**)&d_list, 2*sizeof(hittable*)));
	
	hittable** d_world;
	check_cuda_errors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

	camera** d_camera;
	check_cuda_errors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	create_world<<<1,1>>>(d_list, d_world, d_camera);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	int tx = 8, ty = 8;

	dim3 blocks(IMAGE_WIDTH/tx + 1, IMAGE_HEIGHT/ty + 1);
	dim3 threads(tx, ty);
	std::cerr << "Rendering a " << IMAGE_WIDTH << 'x' << IMAGE_HEIGHT << " image ";
	std::cerr << "in " << tx << 'x' << ty << " blocks.\n";

	rnd_render_init<<<blocks, threads>>>(IMAGE_WIDTH, IMAGE_HEIGHT, d_rand_state);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	clock_t start, stop;

	start = clock();
	render<<<blocks, threads>>>(fb, d_world, d_camera, d_rand_state, IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	stop = clock();

	/* Print in .ppm file */
	double timer = ((double)(stop-start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer << " seconds.\n";

	std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
	color_print print(0.000, 0.999);
	for (int j = IMAGE_HEIGHT - 1; j >= 0; --j) {
		for (int i = 0; i < IMAGE_WIDTH; ++i) {
			size_t pixidx = j*IMAGE_WIDTH + i;
			print.write_color(std::cout, &fb[pixidx]);
		}
	}

	free_world<<<1,1>>>(d_list, d_world, d_camera);
	check_cuda_errors(cudaDeviceSynchronize());
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaFree(d_list));
	check_cuda_errors(cudaFree(d_world));
	check_cuda_errors(cudaFree(d_camera));
	check_cuda_errors(cudaFree(d_rand_state));
	check_cuda_errors(cudaFree(fb));
	cudaDeviceReset();
}