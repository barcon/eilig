#ifndef EILIG_ROUTINES_HPP_
#define EILIG_ROUTINES_HPP_

#include "eilig_vector.hpp"
#include "eilig_matrix.hpp"
#include "eilig_matrix_ellpack.hpp"

#ifdef EILIG_ENABLE_OPENCL
#include "eilig_opencl_vector.hpp"
#include "eilig_opencl_matrix_ellpack.hpp"
#endif

namespace eilig
{
	using CallbackIterative = Status (*)(Status status, Index iteration, Scalar residual);

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
	
	void DecomposeLUP(Matrix& LU, Indices& permutation, const Matrix& A);
	void InverseLUP(Matrix& IA, const Matrix& LU, const Indices& permutation);
	void DiagonalLinearSystem(Vector& x, const Matrix& A, const Vector& b);
	void DiagonalLinearSystem(Vector& x, const Ellpack& A, const Vector& b);
	void ForwardLinearSystem(Vector& x, const Matrix& A, const Vector& b);
	void ForwardLinearSystem(Vector& x, const Ellpack& A, const Vector& b);

	void DirectLUP(Vector& x, const Matrix& LU, const Indices& permutation, const Vector& b);

	Status IterativeJacobi(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0);
	Status IterativeGauss(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0);
	Status IterativeCG(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0);
	Status IterativeBiCGStab(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0);
	void IterativeBiCGStab(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0, CallbackIterative = nullptr);

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

	Status IterativeCG(opencl::Vector& x, const opencl::Ellpack& A, const opencl::Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0);
	Status IterativeBiCGStab(opencl::Vector& x, const opencl::Ellpack& A, const opencl::Vector& b, Scalar rtol = 1.0e-6, Index itmax = 0);

	void WriteToFile(const opencl::Vector& vec, const String& fileName);
	void WriteToFile(const opencl::Ellpack& mat, const String& file);
	
	Status ReadFromFile(opencl::Vector& output, const String& fileName);
	Status ReadFromFile(opencl::Ellpack& output, const String& fileName);

	String ListVector(const opencl::Vector& vector);
	String ListMatrix(const opencl::Ellpack& matrix);

#endif

} /* namespace eilig */

#endif /* EILIG_ROUTINES_HPP_ */