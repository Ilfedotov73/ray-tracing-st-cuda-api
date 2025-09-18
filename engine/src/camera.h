#ifndef CAMERA_H
#define CAMERA_H

class camera
{
private:
	point3 PIXEL_LOC_00;
	vec3   PIXEL_DELTA_U;
	vec3   PIXEL_DELTA_V;

	__device__ void initialize()
	{
		IMAGE_HEIGHT = (IMAGE_HEIGHT == int(IMAGE_HEIGHT / ASPECT_RATIO)) ? 1 : IMAGE_HEIGHT;

		double FOCAL_LENGTH = 1.0;
		double VIEWPORT_HEIGHT = 2.0;
		double VIEWPORT_WIDTH = VIEWPORT_HEIGHT * (double(IMAGE_WIDTH) / IMAGE_HEIGHT);

		vec3 VIEWPORT_U = vec3(VIEWPORT_WIDTH, 0, 0);
		vec3 VIEWPORT_V = vec3(0, -VIEWPORT_HEIGHT, 0);

		PIXEL_DELTA_U = VIEWPORT_U / IMAGE_WIDTH;
		PIXEL_DELTA_V = VIEWPORT_V / IMAGE_HEIGHT;

		point3 VIEWPORT_UPPER_LEFT = CAMERA_CENTER - vec3(0, 0, FOCAL_LENGTH) - VIEWPORT_U / 2 - VIEWPORT_V / 2;
		PIXEL_LOC_00 = VIEWPORT_UPPER_LEFT + 0.5 * (PIXEL_DELTA_U + PIXEL_DELTA_V);
	}
	__device__ color ray_color(const ray& r, hittable** world)
	{
		hit_record rec;
		if ((*world)->hit(r, 0.0, DBL_MAX, rec)) {
			return 0.5 * (rec.normal + color(1, 1, 1));
		}

		vec3 unit_direction = unitv(r.direction());
		double a = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
	}
public:
	double ASPECT_RATIO = 1.0;
	int    IMAGE_WIDTH  = 100;
	int	   IMAGE_HEIGHT = 100;
	
	point3 CAMERA_CENTER;

	__device__ camera() : CAMERA_CENTER(point3(0,0,0)) {}
	__device__ camera(point3 camera_center) : CAMERA_CENTER(camera_center) {}

	__device__ void render(vec3* fb, hittable** world, int i, int j)
	{
		if ((i >= IMAGE_WIDTH) || (j >= IMAGE_HEIGHT)) { return; } 
		
		initialize();
		int pixidx = j*IMAGE_WIDTH + i;
		point3 pixel = PIXEL_LOC_00 + (i*PIXEL_DELTA_U) + (j*PIXEL_DELTA_V);
		vec3 ray_direction = pixel - CAMERA_CENTER;

		ray r(CAMERA_CENTER, ray_direction);
		fb[pixidx] = ray_color(r, world);
	}
};
#endif
