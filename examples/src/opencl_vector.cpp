#include "eilig.hpp"

int main()
{
    try
    {

		eilig::opencl::Initialize("kernels.c", 0, {0, 1});

		auto vector1 = eilig::opencl::Vector({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
		auto vector2 = eilig::opencl::Vector({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
	
		std::cout << eilig::ListVector( vector1 + 1.0) << std::endl;
		std::cout << eilig::ListVector( vector1 - 1.0) << std::endl;
		std::cout << eilig::ListVector(-vector1) << std::endl;
		std::cout << eilig::ListVector( 2.0 * vector1) << std::endl;
		std::cout << eilig::ListVector(vector1.Region(7,8)) << std::endl;
		std::cout << eilig::ListVector(vector1.Region(0,5)) << std::endl;

		auto vector3 = eilig::opencl::Vector(9);
		auto vector4 = eilig::opencl::Vector(9);

		vector4.SetDevice(1);
		vector4(1) = 1.23456;

		vector3 = vector4;
		std::cout << eilig::ListVector(vector3) << std::endl;
		


  //      auto matrix2 = eilig::opencl::Matrix(3, 3);
  //      auto matrix3 = eilig::opencl::Matrix(3, 3, eilig::matrix_diagonal);
  //      auto matrix4 = eilig::opencl::Matrix({ {1.0, 2.0}, {4.0, 5.0} , {7.0, 8.0} });
  //      auto matrix5 = eilig::Matrix({ {1.0, 2.0}, {4.0, 5.0} , {7.0, 8.0} });

  //      matrix2 = 1.23456;

		//std::cout << eilig::ListMatrix(matrix1) << std::endl;
		//std::cout << matrix1.Trace() << std::endl;
		//std::cout << matrix1.Sum() << std::endl;
		//
  //      std::cout << eilig::ListMatrix(matrix1.Region(0, 0, 1, 1)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(0, 0, 0, 2)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(0, 1, 2, 1)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(1, 1, 1, 1)) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Region(2, 1, 1, 1)) << std::endl;

  //      std::cout << eilig::ListMatrix(matrix1.Diagonal()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.DiagonalScale(2.0)) << std::endl;
  //      std::cout << eilig::ListVector(matrix1.DiagonalVector()) << std::endl;
  //      std::cout << eilig::ListMatrix(matrix1.Transpose()) << std::endl;
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