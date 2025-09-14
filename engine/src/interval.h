#ifndef INTERVAL_H
#define INTERVAL_H

class interval
{
public:
	double min, max;
	__device__ interval(double min, double max) : min(min), max(min) {}

	__device__ double size() const { return max - min; }
	__device__ bool contrains(double x) const { return min <= x && x <= max; }
	__device__ bool surrounds(double x) const { return min < x && x < max; }
	__device__ double clip(double x) const
	{
		if (x < min) { return min; }
		if (x > max) { return max; }
		return x;
	}
};

#endif