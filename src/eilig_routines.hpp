#ifndef EILIG_ROUTINES_HPP_
#define EILIG_ROUTINES_HPP_

#include "eilig_status.hpp"
#include "eilig_vector.hpp"
#include "eilig_matrix.hpp"
#include "eilig_matrix_ellpack.hpp"

#ifdef EILIG_ENABLE_OPENCL
	#include "eilig_opencl_kernel.hpp"
	#include "eilig_opencl_vector.hpp"
	#include "eilig_opencl_matrix.hpp"
	//#include "eilig_opencl_matrix_ellpack.hpp"
#endif

namespace eilig
{
	using CallbackIterative = long long int (*)(std::size_t, double);

	inline CallbackIterative callbackIterativeDefault = [](Index iteration, Scalar residual) -> long long int
		{
			Scalar tolerance{ 1e-6 };

			if (std::isnan(residual))
			{
				return EILIG_NOT_CONVERGED;
			}

			if (residual < tolerance)
			{
				return EILIG_SUCCESS;
			}

			return EILIG_CONTINUE;
		};

	Indices CreateIndices();

    Scalar NormMax(const Vector& in); 
	Scalar NormP(const Vector& in, Scalar p);
    Scalar NormP(const Matrix& in, Scalar p);
    Scalar NormP(const Ellpack& in, Scalar p);
	Scalar NormP2(const Vector& in);
	Scalar NormP2(const Matrix& in);
	Scalar NormP2(const Ellpack& in);
    
	Scalar Dot(const Vector& in1, const Vector& in2);
	Vector Cross(const Vector& in1, const Vector& in2);
	
	Scalar Determinant1x1(const Matrix& A);
	Scalar Determinant2x2(const Matrix& A);
	Scalar Determinant3x3(const Matrix& A);
	Matrix Inverse1x1(const Matrix& A);
	Matrix Inverse2x2(const Matrix& A);
	Matrix Inverse3x3(const Matrix& A);
	Matrix ScaleByVector(const Matrix& A, const Vector& alpha);
	
	//void DiagonalLinearSystem(const Matrix& A, Vector& x, const Vector& b);
	//void DiagonalLinearSystem(const Ellpack& A, Vector& x, const Vector& b);
	//void ForwardLinearSystem(const Matrix& A, Vector& x, const Vector& b);
	//void ForwardLinearSystem(const Ellpack& A, Vector& x, const Vector& b);
	//void DecomposeLUP(Matrix& LU, const Matrix& A, Indices& permutation);
	//void InverseLUP(Matrix& IA, const Matrix& LU, const Indices& permutation);
	//void DirectLUP(const Matrix& LU, Vector& x, const Vector& b, const Indices& permutation);
	//void Direct(const Matrix& A, Vector& x, const Vector& b);
	
	Status ReadFromFile(Vector& output, const String& fileName);
	//Status ReadFromFile(Matrix& output, const String& fileName);
	//Status ReadFromFile(Ellpack& output, const String& fileName);

#ifdef EILIG_ENABLE_OPENCL
	Scalar NormMax(const opencl::Vector& in);
	Scalar NormP(const opencl::Vector& in, Scalar p);
	Scalar NormP(const opencl::Matrix& in, Scalar p);
	//Scalar NormP(const opencl::Ellpack& in, Scalar p);
	Scalar NormP2(const opencl::Vector& in);
	Scalar NormP2(const opencl::Matrix& in);
	//Scalar NormP2(const opencl::Ellpack& in);

	Scalar Dot(const opencl::Vector& in1, const opencl::Vector& in2);

#endif

	template <typename T, typename U>
	Status IterativeCG(const T& A, U& x, const U& b, CallbackIterative callbackIterative = nullptr)
	{
		Scalar alpha{ 0.0 };
		Scalar beta{ 0.0 };
		Scalar rho0{ 0.0 };

		Index numberRows = A.GetRows();
		Index iteration = { 0 };

		auto x0 = U(b);
		auto r0 = U(b);
		auto p0 = U(b);
		auto x1 = U(b);
		auto r1 = U(b);

		if (callbackIterative == nullptr)
		{
			callbackIterative = callbackIterativeDefault;
		}

		x0 = x;
		r0 = b - A * x0;

		if (utils::math::IsAlmostEqual(NormP2(r0), 0.0, 5))
		{
			x0 = x + 1.0;
			r0 = b - A * x0;
		}

		p0 = r0;

		for (;;)
		{
			iteration++;

			alpha = Dot(r0, r0) / Dot(p0, A * p0);

			x1 = x0 + alpha * p0;
			r1 = r0 - alpha * (A * p0);

			auto residual = NormP2(r1);
			auto status = callbackIterative(iteration, residual);

			switch (status)
			{
			case EILIG_SUCCESS:
				x = x1;
				return status;
			case EILIG_NOT_CONVERGED:
				x = x1;
				return status;
			case EILIG_STOP:
				x = x1;
				return status;
			case EILIG_CONTINUE:
				break;
			}

			beta = Dot(r1, r1) / Dot(r0, r0);
			p0 = r1 + beta * p0;

			x0 = x1;
			r0 = r1;
		}

		return EILIG_NOT_CONVERGED;
	}

	template <typename T, typename U>
	Status IterativeBiCGStab(const T& A, U& x, const U& b, CallbackIterative callbackIterative = nullptr)
	{
		Scalar alpha{ 0.0 };
		Scalar beta{ 0.0 };
		Scalar omega{ 0.0 };

		Index numberRows = A.GetRows();
		Index iteration{ 0 };

		Vector x0(b);
		Vector r0(b);
		Vector p0(b);
		Vector s0(b);
		Vector h0(b);
		Vector t0(b);
		Vector v0(b);
		Vector x1(b);
		Vector p1(b);
		Vector r1(b);
		Vector r2(b);

		if (callbackIterative == nullptr)
		{
			callbackIterative = callbackIterativeDefault;
		}

		x0 = x;
		r0 = b - A * x0;

		if (utils::math::IsAlmostEqual(NormP2(r0), 0.0, 5))
		{
			x0 = x + 1.0;
			r0 = b - A * x0;
		}

		p0 = r0;
		r1 = r0;

		for (;;)
		{
			iteration++;

			v0 = A * p0;
			alpha = Dot(r1, r0) / Dot(v0, r0);
			h0 = x0 + alpha * p0;
			s0 = r1 - alpha * v0;

			auto residual = NormP2(s0);
			auto status = callbackIterative(iteration, residual);

			switch (status)
			{
			case EILIG_SUCCESS:
				x = h0;
				return status;
			case EILIG_NOT_CONVERGED:
				x = h0;
				return status;
			case EILIG_STOP:
				x = h0;
				return status;
			case EILIG_CONTINUE:
				break;
			}

			t0 = A * s0;
			omega = Dot(t0, s0) / Dot(t0, t0);

			x1 = x0 + alpha * p0 + omega * s0;
			r2 = s0 - omega * t0;

			beta = (Dot(r2, r0) / Dot(r1, r0)) * (alpha / omega);
			p1 = r2 + beta * (p0 - omega * v0);
			r1 = r2;

			p0 = p1;
			x0 = x1;
		}

		return EILIG_NOT_CONVERGED;
	}

	template <typename T>
	String ListVector(const T& vector)
	{
		String output{};
		Index numberRows = vector.GetRows();

		for (Index i = 0; i < numberRows; ++i)
		{
			output += utils::string::Format("{:14.5e}\n", vector.GetValue(i));
		}

		return output;
	}

	template <typename T>
	String ListMatrix(const T& matrix)
	{
		String output{};
		Index numberRows = matrix.GetRows();
		Index numberCols = matrix.GetCols();

		for (Index i = 0; i < numberRows; ++i)
		{
			for (Index j = 0; j < numberCols; ++j)
			{
				output += utils::string::Format("{:14.5e}", matrix.GetValue(i, j));
			}
			output += "\n";
		}
		return output;
	}

	template <typename T>
	Status ReadFromFile(const String& fileName, T& output)
	{
		File file;
		String line;
		Status status;

		file.SetName(fileName);
		file.SetMode(utils::file::Read);

		status = file.Open();
		if (status != utils::file::UTILS_SUCCESS)
		{
			logger::Error(headerEilig, "File could not be opened");
			return EILIG_INVALID_FILE;
		}

		auto stream = static_cast<std::istringstream>(file.GetFull());
		
		if constexpr (std::is_same<T, Vector>::value || std::is_same<T, opencl::Vector>::value)
		{
			Strings table;

			while (std::getline(stream, line))
			{
				if (!utils::string::IsEmpty(line))
				{
					table.push_back(line);
				}
			}

			output.Resize(table.size());

			for (Index i = 0; i < output.GetRows(); i++)
			{
				output(i) = utils::string::ConvertTo<Scalar>(table[i]);
			}
		}
		else if constexpr (std::is_same<T, Matrix>::value || std::is_same<T, Ellpack>::value || std::is_same<T, opencl::Matrix>::value || std::is_same<T, opencl::Ellpack>::value)
		{
			std::vector<Strings> table;
			Strings split;

			while (std::getline(stream, line))
			{
				if (!utils::string::IsEmpty(line))
				{
					split = utils::string::Split(line, { ' ', ';', '\t' });
					table.push_back(split);
				}
			}

			output.Resize(table.size(), table[0].size());

			for (Index i = 0; i < output.GetRows(); i++)
			{
				for (Index j = 0; j < output.GetCols(); j++)
				{
					output.Equal(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
				}
			}
		}

		return EILIG_SUCCESS;
	}

	template <typename T>
	void WriteToFile(const String& fileName, const T& input)
	{
		File file;
		String output;

		file.SetName(fileName);
		file.SetMode(utils::file::Write);

		if (file.Open() != utils::file::UTILS_SUCCESS)
		{
			logger::Error(headerEilig, "File could not be created");
		}

		if constexpr (std::is_same<T, Vector>::value || std::is_same<T, opencl::Vector>::value)
			output = ListVector(input);
		else if constexpr (std::is_same<T, Matrix>::value || std::is_same<T, Ellpack>::value || std::is_same<T, opencl::Matrix>::value || std::is_same<T, opencl::Ellpack>::value)
			output = ListMatrix(input);

		file.Write(output);
	}

} /* namespace eilig */

#endif /* EILIG_ROUTINES_HPP_ */