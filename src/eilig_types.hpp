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
	using Tag = std::size_t;

	enum class MatrixType {
		ones = 1,
		zeros = 2,
		diagonal = 3
	};

	enum class Axis {
		x = 0,
		y = 1,
		z = 2
	};

	class Vector;
	class Matrix;
	class Ellpack;

	using Sparse = Ellpack;
	using Vectors = std::vector<Vector>;
	using Matrices = std::vector<Matrix>;
	using Sparses = std::vector<Sparse>;

	static const String headerEilig = "EILIG";

#ifdef EILIG_ENABLE_OPENCL
	namespace opencl
	{	
		class Vector;
		class Matrix;
		class Ellpack;

		using Sparse = Ellpack;
		using Vectors = std::vector<Vector>;
		using Matrices = std::vector<Matrix>;
		using Sparses = std::vector<Sparse>;
	}
#endif

} /* namespace eilig */

#endif /* EILIG_TYPES_HPP_ */