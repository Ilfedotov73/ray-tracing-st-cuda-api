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
#include "material.h"

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

__device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state, int max_depth)
{
	ray current_ray = r;
	color current_attenuation = color(1.0,1.0,1.0);
	for (int i = 0; i < max_depth; ++i) {
		hit_record rec;
		if ((*world)->hit(current_ray, 0.001, DBL_MAX, rec)) {
			ray	  scattered;
			color attenuation;
			if (rec.mat_ptr->scatter(current_ray, rec, attenuation, scattered, local_rand_state)) {
				current_attenuation *= attenuation;
				current_ray = scattered;
			}
			else {
				return vec3(0.0,0.0,0.0);
			}
		}
		else {
			vec3 unit_direction = unitv(current_ray.direction());
			double a = 0.5 * (unit_direction.y() + 1.0);
			vec3 c = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
			return current_attenuation * c;
		}
	}
	return color(0.0,0.0,0.0);
}

__global__ void rnd_scene_init(curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void rnd_render_init(int imgwidth, int imgheight, curandState* rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= imgwidth) || (j >= imgheight)) { return; }
	int pixidx = j*imgwidth+i;
	curand_init(1984+pixidx, 0, 0, &rand_state[pixidx]);
}

__global__ void render(vec3* fb, hittable** world, camera** cam, curandState* rand_state, int imgwidth, int imgheight,
					   int samples_per_pixel, int max_depth)
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
		ray r = (*cam)->get_ray(u,v, &local_rand_state);
		pixel_color += ray_color(r, world, &local_rand_state, max_depth);
	}
	fb[pixidx] = pixel_color / samples_per_pixel;
}

#define RND (curand_uniform_double(&local_rand_state))
__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, curandState* rand_state, 
							 vec3 lookfrom, vec3 lookat, vec3 vup, double vfov, double aspect_ratio, 
							 double focus_angle, double focus_dist)
{
	if (threadIdx.x == 0 &&  blockIdx.x == 0) {
		d_list[0] = new sphere(point3(0, -1000.0, 01), 1000, new lambertian(color(0.5, 0.5, 0.5)));
		curandState local_rand_state = *rand_state;

		int i = 1;
		for (int a = -11; a < 11; ++a) {
			for (int b = -11; b < 11; ++b) {
				double choose_mat = RND;
				point3 center(a+RND, 0.2, b+RND);
				if (choose_mat < 0.8) {
					d_list[i++] = new sphere(center, 0.2, new lambertian(point3(RND*RND, RND*RND, RND*RND)));
				}
				else if (choose_mat < 0.95) {
					d_list[i++] = new sphere(center, 0.2, new metal(point3(0.5 * (1.0 * RND), 0.5 * (1.0 * RND), 
																	0.5 * (1.0 * RND)), 0.5 * RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
		*d_world  = new hittable_list(d_list, 488);
		*d_camera = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, focus_angle, focus_dist);
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera)
{
	for (int i = 0; i < 488; ++i) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

int main()
{	
	size_t new_stack_size = 2 * 1024;
	check_cuda_errors(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));
	std::cerr << "Set stack size limit: " << new_stack_size << " bytes " << "\n";

	int IMAGE_WIDTH       = 1920,
		IMAGE_HEIGHT      = 1080,
		SAMPLES_PER_PIXEL = 500,
		MAX_DEPTH	      = 50,
		VFOV			  = 20,
		OBJ_COUNTS        = 488;

	point3 LOOKFROM = point3(13,2,3);
	point3 LOOKAT   = point3(0,0,0);
	vec3   VUP		= vec3(0,1,0);

	double ASPECT_RATIO = 16.0 / 8.0,
		   FOCUS_ANGLE  = 0.6,
		   FOCUS_DIST   = 10.0; 

	int num_pixs = IMAGE_WIDTH * IMAGE_HEIGHT;
	size_t fb_size = num_pixs*sizeof(vec3);

	vec3* fb;
	check_cuda_errors(cudaMallocManaged((void**)&fb, fb_size));
	
	curandState* d_rand_state;
	check_cuda_errors(cudaMalloc((void**)&d_rand_state, num_pixs*sizeof(curandState)));
	
	curandState* d_rand_state2;
	check_cuda_errors(cudaMalloc((void**)&d_rand_state2, sizeof(curandState)));

	rnd_scene_init<<<1,1>>>(d_rand_state2);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	/* World */
	hittable** d_list;
	check_cuda_errors(cudaMalloc((void**)&d_list, OBJ_COUNTS*sizeof(hittable*)));
	
	hittable** d_world;
	check_cuda_errors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

	camera** d_camera;
	check_cuda_errors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

	create_world<<<1,1>>>(d_list, d_world, d_camera, d_rand_state2, LOOKFROM, LOOKAT, VUP, VFOV, ASPECT_RATIO, FOCUS_ANGLE, 
						  FOCUS_DIST);
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
	render<<<blocks, threads>>>(fb, d_world, d_camera, d_rand_state, IMAGE_WIDTH, IMAGE_HEIGHT, SAMPLES_PER_PIXEL, 
								MAX_DEPTH);
	check_cuda_errors(cudaGetLastError());
	check_cuda_errors(cudaDeviceSynchronize());

	stop = clock();

	/* Print in .ppm file */
	double timer = ((double)(stop-start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer << " seconds.\n";

	std::cout << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
	for (int j = IMAGE_HEIGHT - 1; j >= 0; --j) {
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
	check_cuda_errors(cudaFree(d_rand_state));
	check_cuda_errors(cudaFree(d_rand_state2));
	check_cuda_errors(cudaFree(fb));
	cudaDeviceReset();
}