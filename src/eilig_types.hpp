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
	using String = utils::String;
	using Strings = utils::Strings;
	using File = utils::file::Text;	
	using Status = long long int;
	using Number = std::size_t;

	static const String headerEilig = "EILIG";

	using Tag = std::size_t;

	using Type = std::size_t;	
	static const Type matrix_ones{ 1 };
	static const Type matrix_zeros{ 2 };
	static const Type matrix_diagonal{ 3 };	

	using Axis = std::size_t;
	static const Axis axis_x{ 0 };
	static const Axis axis_y{ 1 };
	static const Axis axis_z{ 2 };

	class Vector;
	class Matrix;
	class Ellpack;

	using Sparse = Ellpack;
	using Vectors = std::vector<Vector>;
	using Matrices = std::vector<Matrix>;
	using Sparses = std::vector<Sparse>;

#ifdef EILIG_ENABLE_OPENCL
	namespace opencl
	{
		using BufferPtr = std::shared_ptr<club::Buffer>;

		class EntryProxy;
		class Vector;
		class Ellpack;
		
		using PlatformPtr = club::PlatformPtr;
		using ContextPtr = club::ContextPtr;
		using ProgramPtr = club::ProgramPtr;
		using DeviceIndex = club::DeviceIndex;

		class KernelVector;
		using KernelVectorPtr = std::shared_ptr<KernelVector>;

		class KernelMatrix;
		using KernelMatrixPtr = std::shared_ptr<KernelMatrix>;

		class KernelEllpack;
		using KernelEllpackPtr = std::shared_ptr<KernelEllpack>;

		constexpr Index GlobalSize(Index num1, Index num2)
		{
			return num2 * (num1 / num2 + (num1 % num2 != 0));
		}
	}
#endif

} /* namespace eilig */

#endif /* EILIG_TYPES_HPP_ */