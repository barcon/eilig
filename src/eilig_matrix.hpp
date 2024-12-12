#ifndef EILIG_MATRIX_HPP_
#define EILIG_MATRIX_HPP_

#include "eilig_types.hpp"
#include "eilig_vector.hpp"

namespace eilig
{
    class Matrix
    {
    public:
        Matrix();
        Matrix(const Matrix& input);
        Matrix(const Vector& input);
        Matrix(NumberRows numberRows, NumberCols numberCols);
        Matrix(NumberRows numberRows, NumberCols numberCols, Scalar value );
        Matrix(Matrix&& input) noexcept;

        ~Matrix() = default;

        void Resize(NumberRows numberRows, NumberCols numberCols);
        void Resize(NumberRows numberRows, NumberCols numberCols, Scalar value);
        void Fill(Scalar value);

        Scalar operator()(Index row, Index col) const;
        Scalar& operator()(Index row, Index col);

        Scalar operator()(Index index) const;
        Scalar& operator()(Index index);

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
        Matrix Transpose() const;
        Matrix Diagonal() const;
        Matrix Lower(bool diag) const;
        Matrix LowerWithDiagonal() const;
        Matrix LowerWithoutDiagonal() const;
        Matrix Upper(bool diag) const;
        Matrix UpperWithDiagonal() const;
        Matrix UpperWithoutDiagonal() const;
        Matrix Region(Index row1, Index col1, Index row2, Index col2) const;
        void   Region(Index row1, Index col1, Index row2, Index col2, const Matrix& in);

        NumberRows GetRows() const;
        NumberCols GetCols() const;
        Scalar GetValue(Index row, Index col) const;
        const Scalars& GetData() const;

        void SetValue(Index row, Index col, Scalar value);

        friend void Add(Matrix& out, const Matrix& in, Scalar value);
        friend void Add(Matrix& out, const Matrix& in, const Matrix& value);
        friend void Sub(Matrix& out, const Matrix& in, Scalar value);
        friend void Sub(Matrix& out, const Matrix& in, const Matrix& value);
        friend void Mul(Matrix& out, const Matrix& in, Scalar value);
        friend void Mul(Matrix& out, const Matrix& in, const Matrix& value);
        friend void Mul(Vector& out, const Matrix& in, const Vector& value);

    private:
        NumberRows numberRows_{ 0 };
        NumberCols numberCols_{ 0 };
        Scalars data_{};
    };
} /* namespace eilig */

#endif /* EILIG_MATRIX_HPP_ */