#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable 
{
public:
	hittable** objects;
	int objects_count;

	__device__ hittable_list() {}
	__device__ hittable_list(hittable** objects, int objects_count) : objects(objects), 
																	  objects_count(objects_count) {}
	__device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
};

__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const 
{
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = t_max;

	for (int i = 0; i < objects_count; ++i) {
		if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

#endif