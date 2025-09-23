#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable
{
public:
	point3 center;
	double radius;
	material* mat_ptr;

	__device__ sphere() {}
	__device__ sphere(const point3& center, double radius, material* mat_ptr) : center(center), radius(radius), 
					  mat_ptr(mat_ptr) {}
	__device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
};
	__device__ bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
	{
		vec3 oc = center - r.origin();
		double a = r.direction().length_squared();
		double h = dot(r.direction(), oc);
		double c = oc.length_squared() - radius * radius;

		double discriminant = h * h - a * c;
		if (discriminant < 0) { return false; }
		double sqrtd = sqrt(discriminant);

		double root = (h - sqrtd) / a;

		if (!(t_min < root && root < t_max)) {
			root = (h + sqrtd) / a;
			if (!(t_min < root && root < t_max)) { return false; }
		}

		rec.p = r.at(root);
		vec3 outward_normal = (rec.p - center) / radius;
		rec.set_face_normal(r, outward_normal);
		rec.t = root;
		rec.mat_ptr = mat_ptr;

		return true;
	}

#endif
