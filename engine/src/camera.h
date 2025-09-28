#ifndef CAMERA_H
#define CAMERA_H

#define _USE_MATH_DEFINES 
#include <math.h>
#include "ray.h"

__device__ vec3 random_in_unit_disk(curandState* local_rand_state)
{
	for (;;) {
		vec3 p = 2.0*vec3(curand_uniform_double(local_rand_state), curand_uniform_double(local_rand_state),0) - vec3(1,1,0);
		if (p.length_squared() < 1) { return p; }
	}
}

class camera
{
public:	
	point3 PIXEL_LOC_00;
	vec3   PIXEL_DELTA_U;
	vec3   PIXEL_DELTA_V;
	point3 CAMERA_CENTER;
	vec3   U, V, W;

	double FOCUS_DISK;

	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, double vfov, double aspect_ratio, double focus_angle,
					  double focus_dist)
	{
		FOCUS_DISK = focus_angle / 2.0;
		CAMERA_CENTER = lookfrom;
		
		double theta = vfov*M_PI/180;
		double h_height = tan(theta/2);
		double h_width = aspect_ratio * h_height;

		W = unitv(lookfrom - lookat);
		U = unitv(cross(vup, W));
		V = cross(W, U);

		PIXEL_LOC_00 = CAMERA_CENTER - h_width*focus_dist*U - h_height*focus_dist*V - focus_dist*W;
		PIXEL_DELTA_U = 2.0*h_width*focus_dist*U;
		PIXEL_DELTA_V = 2.0*h_height*focus_dist*V;
	}

	__device__ ray get_ray(double u, double v, curandState* local_rand_state)
	{
		point3 pixel_sample = PIXEL_LOC_00 + (u * PIXEL_DELTA_U) + (v * PIXEL_DELTA_V);
		
		vec3 p = FOCUS_DISK * random_in_unit_disk(local_rand_state);
		point3 ray_origin = CAMERA_CENTER + U*p.x() + V*p.y();
		vec3 ray_direction = pixel_sample - ray_origin;
		return ray(ray_origin, ray_direction);
	} 
};
#endif
