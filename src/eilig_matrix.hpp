#ifndef EILIG_MATRIX_HPP_
#define EILIG_MATRIX_HPP_

#include "eilig_types.hpp"
#include "eilig_vector.hpp"
#include "eilig_matrix_ellpack.hpp"

namespace eilig
{
    class Matrix
    {
    public:
        using vector_type = Vector;

        Matrix();
        Matrix(const std::initializer_list<std::initializer_list<Scalar>>& value);
        Matrix(const Matrix& input);
        Matrix(const Matrices& input);
        Matrix(const Ellpack& input);
        Matrix(const Vector& input);
        Matrix(NumberRows numberRows, NumberCols numberCols);
        Matrix(NumberRows numberRows, NumberCols numberCols, Type type);
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
        const Scalars& GetData() const;

        void Equal(Index row, Index col, Scalar value);
        void Equal(const Matrix& value);
        void Equal(const std::initializer_list<std::initializer_list<Scalar>>& value);

        void Add(Scalar value);
        void Add(const Matrix& value);
        void Sub(Scalar value);
        void Sub(const Matrix& value);
        void Mul(Scalar value);
        void Mul(const Matrix& value);

    private:
        NumberRows numberRows_{ 0 };
        NumberCols numberCols_{ 0 };
        Scalars data_{};
    };
} /* namespace eilig */

#endif /* EILIG_MATRIX_HPP_ */