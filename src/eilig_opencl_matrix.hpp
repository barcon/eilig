#ifndef EILIG_OPENCL_MATRIX_HPP_
#define EILIG_OPENCL_MATRIX_HPP_

#include "eilig_types.hpp"
#include "eilig_matrix.hpp"
#include "eilig_matrix_ellpack.hpp"

#include "eilig_opencl_kernel.hpp"
#include "eilig_opencl_entry_proxy.hpp"
#include "eilig_opencl_vector.hpp"

namespace eilig
{
    namespace opencl
    {
        class Matrix
        {
        public:
            Matrix();
            Matrix(const Matrix& input);
            Matrix(const std::initializer_list<std::initializer_list<Scalar>>& value);
            Matrix(const eilig::Matrix& input);
            Matrix(const eilig::Ellpack& input);
            Matrix(NumberRows numberRows, NumberCols numberCols);
            Matrix(NumberRows numberRows, NumberCols numberCols, Type type);
            Matrix(Matrix&& input) noexcept;

            ~Matrix() = default;

            void Resize(NumberRows numberRows, NumberCols numberCols);
            void Resize(NumberRows numberRows, NumberCols numberCols, Scalar value);
            void Fill(Scalar value);

            EntryProxy operator()(Index row, Index col);

            Matrix& operator=(Scalar rhs);
            Matrix& operator=(const Matrix& rhs);
            Matrix& operator=(Matrix&& rhs) noexcept;

            Matrix operator+(Scalar rhs) const;
            Matrix operator+(const Matrix& rhs) const;
            Matrix operator+() const;
            friend Matrix operator+(Scalar lhs, const Matrix& rhs);

            Matrix operator-(Scalar rhs) const;
            Matrix operator-(const Matrix& rhs) const;
            Matrix operator-() const;
            friend Matrix operator-(Scalar lhs, const Matrix& rhs);

            Matrix operator*(Scalar rhs) const;
            Matrix operator*(const Matrix& rhs) const;
            Vector operator*(const Vector& rhs) const;
            friend Matrix operator*(Scalar lhs, const Matrix& rhs);

            Matrix& SwapRows(Index row1, Index row2);
            Matrix& SwapCols(Index cols1, Index cols2);
            Scalar Trace() const;
            Scalar Sum() const;
            Matrix Transpose() const;
            Matrix Diagonal() const;
            Matrix DiagonalScale(Scalar factor) const;
            Vector DiagonalVector() const;
            Matrix Lower(bool diag) const;
            Matrix LowerWithDiagonal() const;
            Matrix LowerWithoutDiagonal() const;
            Matrix Upper(bool diag) const;
            Matrix UpperWithDiagonal() const;
            Matrix UpperWithoutDiagonal() const;
            Matrix Region(Index row1, Index col1, Index row2, Index col2) const;
            void   Replace(Index row1, Index col1, const Matrix& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            Scalar GetValue(Index row, Index col) const;
            BufferPtr GetDataGPU() const;
            KernelMatrixPtr GetKernel() const;
            const DeviceIndex& GetDeviceIndex() const;

            void Equal(Index row, Index col, Scalar value);
            void Equal(const Matrix& value);
            void Equal(const std::initializer_list<std::initializer_list<Scalar>>& value);

            void SetDevice(DeviceIndex deviceIndex);

        private:
            void InitKernel();

            NumberRows numberRows_{ 0 };
            NumberCols numberCols_{ 0 };

            DeviceIndex deviceIndex_{ 0 };

            KernelMatrixPtr kernel_{ nullptr };
            BufferPtr dataGPU_{ nullptr };
        };
    }
} /* namespace eilig */

#endif /* EILIG_OPENCL_MATRIX_HPP_ */