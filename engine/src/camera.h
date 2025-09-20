#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class camera
{
public:	
	point3 PIXEL_LOC_00;
	vec3   PIXEL_DELTA_U;
	vec3   PIXEL_DELTA_V;
	point3 CAMERA_CENTER;

	__device__ camera()
	{
		CAMERA_CENTER = point3(0,0,0);
		PIXEL_LOC_00  = point3(-2.0,-1.0,-1.0);
		PIXEL_DELTA_U = vec3(4.0, 0.0, 0.0);
		PIXEL_DELTA_V = vec3(0.0, 2.0, 0.0);
	}

	__device__ ray get_ray(double u, double v)
	{
		point3 pixel_sample = PIXEL_LOC_00 + (u * PIXEL_DELTA_U) + (v * PIXEL_DELTA_V);
		vec3 ray_direction = pixel_sample - CAMERA_CENTER;
		return ray(CAMERA_CENTER, ray_direction);
	} 
};
#endif
