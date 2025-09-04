#ifndef COLOR_H
#define COLOR_H

#include <iostream>
using color = vec3;

void write_color(std::ostream& out, void* pxcolor)
{
	double r = ((color*)pxcolor)->x();
	double g = ((color*)pxcolor)->y();
	double b = ((color*)pxcolor)->z();

	int rbyte = int(255.999 * r);
	int gbyte = int(255.999 * g);
	int bbyte = int(255.999 * b);

	out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
