#include "eilig_matrix.hpp"

namespace eilig
{
    Matrix::Matrix()
    {
        Resize(1, 1);
    }
    Matrix::Matrix(Matrix&& input) noexcept
    {
        (*this) = std::move(input);
    }
    Matrix::Matrix(const Matrix& input)
    {
        (*this) = input;
    }
    Matrix::Matrix(const Ellpack& input)
    {
		Resize(input.GetRows(), input.GetCols());

        const auto& count = input.GetCount();
        const auto& position = input.GetPosition();
        const auto& data = input.GetData();

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index k = 0; k < count[i]; ++k)
            {
                auto j = position[];
                (*this)(i, j) = data[i][j];
            }
		}   
    }
    Matrix::Matrix(const Vector& input)
    {
        numberRows_ = input.GetRows();
        numberCols_ = input.GetCols();
        data_ = input.GetData();
    }
    Matrix::Matrix(NumberRows numberRows, NumberRows numberCols)
    {
        Resize(numberRows, numberCols);
    }
    Matrix::Matrix(NumberRows numberRows, NumberRows numberCols, Type type)
    {
        switch (type)
        {
        case matrix_ones:
            Resize(numberRows, numberCols, 1.0);
            break;
        case matrix_diagonal:
            Resize(numberRows, numberCols, 0.0);

            for (Index i = 0; (i < numberRows) && (i < numberCols); ++i)
            {
                (*this)(i, i) = 1.0;
            }
            break;
        case matrix_zeros:
        default:
            Resize(numberRows, numberCols, 0.0);
        }
    }
    void Matrix::Resize(NumberRows numberRows, NumberRows numberCols)
    {
        numberRows_ = numberRows;
        numberCols_ = numberCols;
        data_ = Scalars(numberRows_ * numberCols_, 0.0);
    }
    void Matrix::Resize(NumberRows numberRows, NumberRows numberCols, Scalar value)
    {
        Resize(numberRows, numberCols);
        Fill(value);
    }
    void Matrix::Fill(Scalar value)
    {
        (*this) = value;
    }
    Scalar Matrix::operator()(Index row, Index col) const
    {
        return data_[row * numberCols_ + col];
    }
    Scalar& Matrix::operator()(Index row, Index col)
    {
        return data_[row * numberCols_ + col];
    }
    Scalar Matrix::operator()(Index index) const
    {
        return data_[index];
    }
    Scalar& Matrix::operator()(Index index)
    {
        return data_[index];
    }
    Matrix& Matrix::operator=(Scalar rhs)
    {
        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                data_[i * numberCols_ + j] = rhs;
            }
        }

        return *this;
    }
    Matrix& Matrix::operator=(Matrix&& rhs) noexcept
    {
        if (&rhs == this)
        {
            return *this;
        }

        numberRows_ = rhs.numberRows_;
        numberCols_ = rhs.numberCols_;
        data_ = Scalars(std::move(rhs.data_));

        return *this;
    }
    Matrix& Matrix::operator=(const Matrix& rhs)
    {
        numberRows_ = rhs.numberRows_;
        numberCols_ = rhs.numberCols_;
        data_ = rhs.data_;

        return *this;
    }
    Matrix Matrix::operator+(Scalar rhs) const
    {
        Matrix res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                res.data_[i * numberCols_ + j] += rhs;
            }
        }

        return res;
    }
    Matrix Matrix::operator+(const Matrix& rhs) const
    {
        Matrix res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                res.data_[i * numberCols_ + j] += rhs.data_[i * numberCols_ + j];
            }
        }

        return res;
    }
    Matrix Matrix::operator+() const
    {
        return (*this);
    }
    Matrix operator+(Scalar lhs, const Matrix& rhs)
    {
        return rhs + lhs;
    }
    Matrix Matrix::operator-(Scalar rhs) const
    {
        Matrix res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                res.data_[i * numberCols_ + j] -= rhs;
            }
        }

        return res;
    }
    Matrix Matrix::operator-(const Matrix& rhs) const
    {
        Matrix res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                res.data_[i * numberCols_ + j] -= rhs.data_[i * numberCols_ + j];
            }
        }

        return res;
    }
    Matrix Matrix::operator-() const
    {
        return -1.0 * (*this);
    }
    Matrix operator-(Scalar lhs, const Matrix& rhs)
    {
        return -rhs + lhs;
    }
    Matrix Matrix::operator*(Scalar rhs) const
    {
        Matrix res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                res.data_[i * numberCols_ + j] *= rhs;
            }
        }

        return res;
    }
    Matrix Matrix::operator*(const Matrix& rhs) const
    {
        Matrix res(numberRows_, rhs.numberCols_, 0.0);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index k = 0; k < numberCols_; ++k)
            {
                for (Index j = 0; j < rhs.numberCols_; ++j)
                {
                    res(i, j) += (*this)(i, k) * rhs(k, j);
                }
            }
        }

        return res;
    }
    Vector Matrix::operator*(const Vector& rhs) const
    {
        Vector res(numberRows_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res(i) = 0.0;

            for (Index k = 0; k < numberCols_; ++k)
            {
                res(i) += (*this)(i, k) * rhs(k);
            }
        }

        return res;
    }
    Matrix operator*(Scalar lhs, const Matrix& rhs)
    {
        return rhs * lhs;
    }
    Matrix& Matrix::SwapRows(Index row1, Index row2)
    {
        Scalar temp;

        for (Index j = 0; j < numberCols_; ++j)
        {
            temp = data_[row2 * numberCols_ + j];
            data_[row2 * numberCols_ + j] = data_[row1 * numberCols_ + j];
            data_[row1 * numberCols_ + j] = temp;
        }

        return *this;
    }
    Matrix& Matrix::SwapCols(Index col1, Index col2)
    {
        Scalar temp;

        for (Index i = 0; i < numberRows_; ++i)
        {
            temp = data_[i * numberCols_ + col2];
            data_[i * numberCols_ + col2] = data_[i * numberCols_ + col1];
            data_[i * numberCols_ + col1] = temp;
        }

        return *this;
    }
    Matrix Matrix::Transpose() const
    {
        Matrix res(numberCols_, numberRows_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                res(j, i) = (*this)(i, j);
            }
        }

        return res;
    }
    Matrix Matrix::Diagonal() const
    {
        Matrix res(numberRows_, numberCols_, 0.0);

        for (Index i = 0; (i < numberRows_) && (i < numberCols_); ++i)
        {
            res(i, i) = (*this)(i, i);
        }

        return res;
    }
    Matrix Matrix::Lower(bool diag) const
    {
        if (diag)
        {
            return LowerWithDiagonal();
        }
        else
        {
            return LowerWithoutDiagonal();
        }
    }
    Matrix Matrix::LowerWithDiagonal() const
    {
        Matrix res(numberRows_, numberCols_, 0.0);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                if (j <= i)
                {
                    res(i, j) = (*this)(i, j);
                }
                else
                {
                    break;
                }
            }
        }

        return res;
    }
    Matrix Matrix::LowerWithoutDiagonal() const
    {
        Matrix res(numberRows_, numberCols_, 0.0);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < numberCols_; ++j)
            {
                if (j < i)
                {
                    res(i, j) = (*this)(i, j);
                }
                else
                {
                    break;
                }
            }
        }

        return res;
    }
    Matrix Matrix::Upper(bool diag) const
    {
        if (diag)
        {
            return UpperWithDiagonal();
        }
        else
        {
            return UpperWithoutDiagonal();
        }
    }
    Matrix Matrix::UpperWithDiagonal() const
    {
        Matrix res(numberRows_, numberCols_, 0.0);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = numberCols_ - 1; j >= 0; --j)
            {
                if (j >= i)
                {
                    res(i, j) = (*this)(i, j);
                }
                else
                {
                    break;
                }
            }
        }

        return res;
    }
    Matrix Matrix::UpperWithoutDiagonal() const
    {
        Matrix res(numberRows_, numberCols_, 0.0);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = numberCols_ - 1; j >= 0; --j)
            {
                if (j > i)
                {
                    res(i, j) = (*this)(i, j);
                }
                else
                {
                    break;
                }
            }
        }

        return res;
    }
    Matrix Matrix::Region(Index row1, Index col1, Index row2, Index col2) const
    {
        Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
        Index aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
        Index aux3;
        Index aux4;
        Matrix res(aux1, aux2);

        if ((row1 <= row2) && (col1 <= col2))
        {
            aux3 = row1;
            aux4 = col1;
        }
        else if ((row1 >= row2) && (col1 <= col2))
        {
            aux3 = row2;
            aux4 = col1;
        }
        else if ((row1 >= row2) && (col1 >= col2))
        {
            aux3 = row2;
            aux4 = col2;
        }
        else
        {
            aux3 = row1;
            aux4 = col2;
        }

        for (Index i = 0; i < aux1; ++i)
        {
            for (Index j = 0; j < aux2; ++j)
            {
                res(i, j) = (*this)(aux3 + i, aux4 + j);
            }
        }

        return res;
    }
    void   Matrix::Region(Index row1, Index col1, Index row2, Index col2, const Matrix& in)
    {
        Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
        Index aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
        Index aux3;
        Index aux4;

        if ((row1 <= row2) && (col1 <= col2))
        {
            aux3 = row1;
            aux4 = col1;
        }
        else if ((row1 >= row2) && (col1 <= col2))
        {
            aux3 = row2;
            aux4 = col1;
        }
        else if ((row1 >= row2) && (col1 >= col2))
        {
            aux3 = row2;
            aux4 = col2;
        }
        else
        {
            aux3 = row1;
            aux4 = col2;
        }

        for (Index i = 0; i < aux1; ++i)
        {
            for (Index j = 0; j < aux2; ++j)
            {
                (*this)(aux3 + i, aux4 + j) = in(i, j);
            }
        }
    }
    NumberRows Matrix::GetRows() const
    {
        return numberRows_;
    }
    NumberCols Matrix::GetCols() const
    {
        return numberCols_;
    }
    Scalar Matrix::GetValue(Index row, Index col) const
    {
        return data_[row * numberCols_ + col];
    }
    const Scalars& Matrix::GetData() const
    {
        return data_;
    }
    void Matrix::SetValue(Index row, Index col, Scalar value)
    {
        (*this)(row, col) = value;
    }
    void Add(Matrix& out, const Matrix& in, Scalar value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            for (Index j = 0; j < cols; ++j)
            {
                out(i, j) = in(i, j) + value;
            }
        }
    }
    void Add(Matrix& out, const Matrix& in, const Matrix& value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            for (Index j = 0; j < cols; ++j)
            {
                out(i, j) = in(i, j) + value(i, j);
            }
        }
    }
    void Sub(Matrix& out, const Matrix& in, Scalar value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            for (Index j = 0; j < cols; ++j)
            {
                out(i, j) = in(i, j) - value;
            }
        }
    }
    void Sub(Matrix& out, const Matrix& in, const Matrix& value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            for (Index j = 0; j < cols; ++j)
            {
                out(i, j) = in(i, j) - value(i, j);
            }
        }
    }
    void Mul(Matrix& out, const Matrix& in, Scalar value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            for (Index j = 0; j < cols; ++j)
            {
                out(i, j) = in(i, j) * value;
            }
        }
    }
    void Mul(Matrix& out, const Matrix& in, const Matrix& value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();
        auto aux = value.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            for (Index j = 0; j < aux; ++j)
            {
                out(i, j) = 0.0;
            }

            for (Index k = 0; k < cols; ++k)
            {
                for (Index j = 0; j < aux; ++j)
                {
                    out(i, j) += in(i, k) * value(k, j);
                }
            }
        }
    }
    void Mul(Vector& out, const Matrix& in, const Vector& value)
    {
        auto rows = in.GetRows();
        auto cols = in.GetCols();

        for (Index i = 0; i < rows; ++i)
        {
            out(i) = 0.0;

            for (Index k = 0; k < cols; ++k)
            {
                out(i) += in(i, k) * value(k);
            }
        }
    }
} /* namespace eilig */