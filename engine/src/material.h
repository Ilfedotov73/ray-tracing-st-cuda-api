#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"

#define RANDVEC3 vec3(curand_uniform_double(local_rand_state),curand_uniform_double(local_rand_state),curand_uniform_double(local_rand_state))
__device__ vec3 random_unit_vector(curandState* local_rand_state)
{
	for (;;) {
		vec3 p = 2.0 * RANDVEC3 - vec3(1, 1, 1);
		double lensq = p.length();
		if (1e-160 < lensq && lensq <= 1) { return p / sqrt(lensq); }
	}
}
__device__ vec3 reflect(const vec3& v, const vec3& n) { return v - 2*dot(v,n)*n; }
__device__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) 
{
	double cos_theta = fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}
__device__ bool near_zero(const vec3& e)
{
	double s = 1e-8;
	return(fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}
__device__ double reflectance(double cosine, double refraction_index)
{
	double r0 = (1 - refraction_index) / (1 + refraction_index);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}

class material
{
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
									curandState* local_rand_state) const = 0;
};

class lambertian : public material
{
public:
	color albedo;

	__device__ lambertian(const color& albedo) : albedo(albedo) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
									curandState* local_rand_state) const
	{
		vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);
		if (near_zero(scatter_direction)) { scatter_direction = rec.normal; }
		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
	}
};

class metal : public material
{
public:
	color albedo;
	double fuzz;
	__device__ metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz:1) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
									curandState* local_rand_state) const
	{
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		reflected = unitv(reflected) + (fuzz * random_unit_vector(local_rand_state));
		scattered = ray(rec.p, reflected);
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
};

class dielectric : public material
{
public:
	double refraction_index;
	__device__ dielectric(double refraction_index) : refraction_index(refraction_index) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
		curandState* local_rand_state) const
	{
		attenuation = color(1.0,1.0,1.0);
		double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

		vec3 unit_direction = unitv(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

		bool cannot_refract = ri * sin_theta > 1.0;

		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform_double(local_rand_state)) { 
			direction = reflect(unit_direction, rec.normal); 
		}
		else { direction = refract(unit_direction, rec.normal, ri); }

		scattered = ray(rec.p, direction);
		return true;
	}
};

#endif
