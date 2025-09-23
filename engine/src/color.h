#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using color = vec3;

inline double liner_to_gamma(double liner_component)
{
	if (liner_component > 0) { return sqrt(liner_component); }
	return 0;
}

double clamp(double x, double min, double max)
{
	if (x < min) { return min; }
	if (x > max) { return max; }
	return x;
}

void write_color(std::ostream& out, void* pxcolor)
{
	double min = 0.000,
		   max = 0.999;

	double r = ((color*)pxcolor)->x();
	double g = ((color*)pxcolor)->y();
	double b = ((color*)pxcolor)->z();

	r = liner_to_gamma(r);
	g = liner_to_gamma(g);
	b = liner_to_gamma(b);

	int rbyte = int(256 * clamp(r, min, max));
	int gbyte = int(256 * clamp(g, min, max));
	int bbyte = int(256 * clamp(b, min, max));

	out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
