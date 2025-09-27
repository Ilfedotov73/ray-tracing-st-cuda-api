#ifndef CAMERA_H
#define CAMERA_H

#define _USE_MATH_DEFINES 
#include <math.h>
#include "ray.h"

class camera
{
public:	
	point3 PIXEL_LOC_00;
	vec3   PIXEL_DELTA_U;
	vec3   PIXEL_DELTA_V;
	point3 CAMERA_CENTER;

	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, double vfov, double aspect_ratio)
	{
		vec3 u, v, w;
		double theta = vfov*M_PI/180;
		double h_height = tan(theta/2);
		double h_width = aspect_ratio * h_height;
		CAMERA_CENTER = lookfrom;

		w = unitv(lookfrom - lookat);
		u = unitv(cross(vup, w));
		v = cross(w, u);

		PIXEL_LOC_00 = CAMERA_CENTER - h_width*u - h_height*v - w;
		PIXEL_DELTA_U = 2*h_width*u;
		PIXEL_DELTA_V = 2*h_height*v;
	}

	__device__ ray get_ray(double u, double v)
	{
		point3 pixel_sample = PIXEL_LOC_00 + (u * PIXEL_DELTA_U) + (v * PIXEL_DELTA_V);
		vec3 ray_direction = pixel_sample - CAMERA_CENTER;
		return ray(CAMERA_CENTER, ray_direction);
	} 
};
#endif
