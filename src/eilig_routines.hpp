#ifndef EILIG_ROUTINES_HPP_
#define EILIG_ROUTINES_HPP_

#include "eilig_vector.hpp"
#include "eilig_matrix.hpp"
#include "eilig_matrix_ellpack.hpp"

#ifdef EILIG_ENABLE_OPENCL
#include "eilig_opencl_vector.hpp"
#include "eilig_opencl_matrix_ellpack.hpp"
#endif

using CallbackIterative = long long int (*)(std::size_t, double);

namespace eilig
{
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
	
	Scalar DeterminantLUP(const Matrix& LU, const Indices& permutation);
	Scalar Determinant(const Matrix& A);
	Matrix Inverse(const Matrix& A);
	Matrix ScaleByVector(const Matrix& A, const Vector& alpha);
	Vector Solve(const Matrix& A, const Vector& b);
	
	void DiagonalLinearSystem(const Matrix& A, Vector& x, const Vector& b);
	void DiagonalLinearSystem(const Ellpack& A, Vector& x, const Vector& b);
	void ForwardLinearSystem(const Matrix& A, Vector& x, const Vector& b);
	void ForwardLinearSystem(const Ellpack& A, Vector& x, const Vector& b);
	void DecomposeLUP(Matrix& LU, const Matrix& A, Indices& permutation);
	void InverseLUP(Matrix& IA, const Matrix& LU, const Indices& permutation);
	void DirectLUP(const Matrix& LU, Vector& x, const Vector& b, const Indices& permutation);
	Status IterativeCG(const Ellpack& A, Vector& x, const Vector& b, CallbackIterative callbackIterative);
	Status IterativeBiCGStab(const Ellpack& A, Vector& x, const Vector& b, CallbackIterative callbackIterative);

	void WriteToFile(const Vector& vec, const String& fileName);
	void WriteToFile(const Matrix& mat, const String& fileName);
	void WriteToFile(const Ellpack& mat, const String& fileName);

	Status	ReadFromFile(Vector& output, const String& fileName);
	Status	ReadFromFile(Matrix& output, const String& fileName);
	Status	ReadFromFile(Ellpack& output, const String& fileName);

	String ListVector(const Vector& vector);
	String ListMatrix(const Matrix& matrix);
	String ListMatrix(const Ellpack& matrix);

#ifdef EILIG_ENABLE_OPENCL
	Scalar NormMax(const opencl::Vector& in);
	Scalar NormP(const opencl::Vector& in, Scalar p);
	Scalar NormP(const opencl::Ellpack& in, Scalar p);
	Scalar NormP2(const opencl::Vector& in);
	Scalar NormP2(const opencl::Ellpack& in);
	Scalar Dot(const opencl::Vector& in1, const opencl::Vector& in2);
	Status IterativeCG(const opencl::Ellpack& A, opencl::Vector& x, const opencl::Vector& b, CallbackIterative callbackIterative);
	Status IterativeBiCGStab(const opencl::Ellpack& A, opencl::Vector& x, const opencl::Vector& b, CallbackIterative callbackIterative);

	void WriteToFile(const opencl::Vector& vec, const String& fileName);
	void WriteToFile(const opencl::Ellpack& mat, const String& file);
	
	Status ReadFromFile(opencl::Vector& output, const String& fileName);
	Status ReadFromFile(opencl::Ellpack& output, const String& fileName);

	String ListVector(const opencl::Vector& vector);
	String ListMatrix(const opencl::Ellpack& matrix);

#endif

} /* namespace eilig */

#endif /* EILIG_ROUTINES_HPP_ */