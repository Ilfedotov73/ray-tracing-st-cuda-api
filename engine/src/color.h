#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using color = vec3;

class color_print
{
public:
	double min, max;

	color_print() {}
	color_print(double min, double max) : min(min), max(max) {}

	double clamp(double x) 
	{
		if (x < min) { return min; }
		if (x > max) { return max; }
		return x;
	}

	void write_color(std::ostream& out, void* pxcolor)
	{
		double r = ((color*)pxcolor)->x();
		double g = ((color*)pxcolor)->y();
		double b = ((color*)pxcolor)->z();

		int rbyte = int(256 * clamp(r));
		int gbyte = int(256 * clamp(g));
		int bbyte = int(256 * clamp(b));

		out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
	}

};

#endif
