#ifndef VEC3_H
#define VEC3_H

class vec3
{
public:
	double e[3];
	__host__ __device__ vec3() : e{ 0.0, 0.0, 0.0 } {}
	__host__ __device__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}
	__host__ __device__ double x() const { return e[0]; }
	__host__ __device__ double y() const { return e[1]; }
	__host__ __device__ double z() const { return e[2]; }

	__host__ __device__ vec3    operator-() const { return vec3( -e[0], -e[1], -e[2]); }
	__host__ __device__ double  operator[](int i) const { return e[i]; }
	__host__ __device__ double& operator[](int i) { return e[i]; }
	
	__host__ __device__ vec3& operator+=(const vec3& v)
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}
	__host__ __device__ vec3& operator*=(double t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}
	__host__ __device__ vec3& operator/=(double t) { return *this *= 1/t;}

	__host__ __device__ double length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2] * e[2]; }
	__host__ __device__ double length() const { return std::sqrt(length_squared()); }
	__host__ __device__ static vec3 random() { return vec3(random_double(), random_double(), random_double()); }
	__host__ __device__ static vec3 radnom(double min, double max) { return vec3(random_double(min, max), 
																				 random_double(min, max),
																			   	 random_double(min, max)); }
	__host__ __device__ bool near_zero() const
	{
		double s = 1e-8;
		return(std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
	}
};

using point3 = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) { return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2]; }

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
	return vec3(
		u.e[0] + v.e[0],
		u.e[1] + v.e[1],
		u.e[2] + v.e[2]
	);
}
__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
	return vec3(
		u.e[0] - v.e[0],
		u.e[1] - v.e[1],
		u.e[2] - v.e[2]
	);
}
__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v)
{
	return vec3(
		u.e[0] * v.e[0],
		u.e[1] * v.e[1],
		u.e[2] * v.e[2]
	);
}
__host__ __device__ inline vec3 operator*(double t, const vec3& v)
{
	return vec3(
		t*v.e[0],
		t*v.e[1],
		t*v.e[2]
	);
}
__host__ __device__ inline vec3 operator*(const vec3& v, double t) { return t*v; }
__host__ __device__ inline vec3 operator/(const vec3& v, double t) { return (1/t)*v; }

__host__ __device__ inline double dot(const vec3& u, const vec3& v)
{
	return u.e[0] * v.e[0]
		 + u.e[1] * v.e[1]
		 + u.e[2] * v.e[2];
}
__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
				u.e[2] * v.e[0] - u.e[0] * v.e[2],
				u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}
__host__ __device__ inline vec3 unitv(const vec3& v) { return v/v.length(); }
__host__ __device__ inline vec3 random_unitv()
{
	for (;;) {
		vec3 p = vec3::radnom(-1,1);
		double lensq = p.length_squared();
		if (1e-160 < lensq && lensq <= 1) { return p/sqrt(lensq); }
	}
}
__host__ __device__ inline vec3 random_on_hemisphere(const vec3& normal)
{
	vec3 rndv_in_unit_sp = random_unitv();
	if (dot(rndv_in_unit_sp, normal) > 0.0) { return rndv_in_unit_sp; }
	else { return -rndv_in_unit_sp; }
}
__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) { return v - 2*dot(v,n)*n; }
__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
	double cos_theta = std::fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp  = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_paral = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared()))*n;
	return r_out_perp + r_out_paral;
}
__host__ __device__ inline vec3 random_in_unit_disk()
{
	for (;;) {
		vec3 p = vec3(random_double(-1,1), random_double(-1,1), 0);
		if (p.length_squared() < 1) { return p; }
	}
}
#endif