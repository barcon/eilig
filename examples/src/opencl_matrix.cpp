#include "eilig.hpp"

int main()
{
    try
    {

		eilig::opencl::Initialize("kernels.c", 0, { 0 });

        auto matrix1 = eilig::opencl::Matrix({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} , {7.0, 8.0, 9.0} });
        auto matrix2 = eilig::opencl::Matrix({ {1.0, 2.0}, {4.0, 5.0} , {7.0, 8.0} });
        auto matrix3 = eilig::opencl::Matrix(3, 3, eilig::matrix_diagonal);
		auto vector1 = eilig::opencl::Vector({1.0, 2.0, 3.0});
  //      auto matrix4 = eilig::opencl::Matrix({ {1.0, 2.0}, {4.0, 5.0} , {7.0, 8.0} });

  //      matrix2 = 1.23456;

		std::cout << matrix1.Trace() << std::endl;
		std::cout << matrix1.Sum() << std::endl;

		std::cout << eilig::ListMatrix(matrix1) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.Transpose()) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.Diagonal()) << std::endl;
		std::cout << eilig::ListVector(matrix1.DiagonalVector()) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.LowerWithDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.LowerWithoutDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.UpperWithDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.UpperWithoutDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.Region(1, 1, 1, 1)) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.Region(0, 0, 1, 1)) << std::endl;
		std::cout << eilig::ListMatrix(matrix1.Region(1, 1, 2, 2)) << std::endl;

		std::cout << "-----------------------------------------------------------------" << std::endl;

		std::cout << eilig::ListMatrix(matrix2) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.Transpose()) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.Diagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.DiagonalScale(2.0)) << std::endl;
		std::cout << eilig::ListVector(matrix2.DiagonalVector()) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.LowerWithDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.LowerWithoutDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.UpperWithDiagonal()) << std::endl;
		std::cout << eilig::ListMatrix(matrix2.UpperWithoutDiagonal()) << std::endl;

		std::cout << "-----------------------------------------------------------------" << std::endl;
		std::cout << eilig::ListMatrix(matrix3) << std::endl;
		std::cout << eilig::ListMatrix(matrix1 * matrix3) << std::endl;
		std::cout << eilig::ListVector(matrix1 * vector1) << std::endl;
		//
  //      std::cout << eilig::ListMatrix(matrix1.Region(0, 0, 1, 1)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(0, 0, 0, 2)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(0, 1, 2, 1)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(1, 1, 1, 1)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(2, 1, 1, 1)) << std::endl;

  //      std::cout << eilig::ListMatrix(matrix1.Diagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.DiagonalScale(2.0)) << std::endl;
  //      std::cout << eilig::ListVector(matrix1.DiagonalVector()) << std::endl;

  //      std::cout << eilig::ListMatrix(matrix1.LowerWithDiagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.UpperWithDiagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.SwapRows(1, 2)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.SwapCols(0, 1)) << std::endl;

  //      std::cout << eilig::ListMatrix(matrix4.Diagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix4.DiagonalScale(2.0)) << std::endl;
  //      std::cout << eilig::ListVector(matrix4.DiagonalVector()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix4.Transpose()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix4.LowerWithDiagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix4.UpperWithDiagonal()) << std::endl;

  //      std::cout << eilig::ListMatrix(matrix5.Diagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix5.DiagonalScale(2.0)) << std::endl;
  //      std::cout << eilig::ListVector(matrix5.DiagonalVector()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix5.Transpose()) << std::endl;

        //std::cout << eilig::ListMatrix(matrix2) << std::endl;
		//std::cout << eilig::ListMatrix(matrix3) << std::endl;
		//std::cout << eilig::ListMatrix(1.0 + matrix1) << std::endl;
		//std::cout << eilig::ListMatrix(1.0 - matrix1) << std::endl;
		//std::cout << eilig::ListMatrix(2.0 * matrix1) << std::endl;
		//std::cout << eilig::ListMatrix(matrix1 + matrix1) << std::endl;
		//std::cout << eilig::ListMatrix(matrix1 + matrix2) << std::endl;
		//
        //std::cout << eilig::ListMatrix(matrix4.Transpose()) << std::endl;
    }
    catch (const std::exception& error)
    {
        logger::Error("EXAMPLE", error.what());
    }

    return 0;
}