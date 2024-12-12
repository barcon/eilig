#ifndef EILIG_TYPES_HPP_
#define EILIG_TYPES_HPP_

#include "utils.hpp"
#include "logger.hpp"

#ifdef EILIG_ENABLE_OPENCL
#include "club.hpp"
#endif

#include <cassert>
#include <vector>

namespace eilig
{
	using Scalar = double;
	using Scalars = std::vector<Scalar>;
	using NumberRows = std::size_t;
	using NumberCols = std::size_t;
	using Index = std::size_t;
	using Indices = std::vector<Index>;
	using String = std::string;
	using File = utils::file::Text;
	using Status = long long int;

	static const String headerEilig = "EILIG";

	using Axis = std::size_t;
	static const Axis axis_x{ 0 };
	static const Axis axis_y{ 1 };
	static const Axis axis_z{ 2 };

	class Vector;
	class Matrix;
	class Ellpack;

	using Vectors = std::vector<Vector>;
	using Matrices = std::vector<Matrix>;
	using Ellpacks = std::vector<Ellpack>;

#ifdef EILIG_ENABLE_OPENCL
	namespace opencl
	{
		class Kernels;
		using KernelsPtr = Kernels*;
		using ConstKernelsPtr = const Kernels*;

		class EntryProxy;
		class Vector;
		class Ellpack;

		constexpr Index GlobalSize(Index num1, Index num2)
		{
			return num2 * (num1 / num2 + (num1 % num2 != 0));
		}
	}
#endif

} /* namespace eilig */

#endif /* EILIG_TYPES_HPP_ */