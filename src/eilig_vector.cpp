#include "eilig_vector.hpp"

namespace eilig
{
    Vector::Vector()
    {
        Resize(1);
    }
    Vector::Vector(const Scalars& values)
    {
        data_ = values;
        numberRows_ = data_.size();
    }
    Vector::Vector(Vector&& input) noexcept
    {
        (*this) = std::move(input);
    }
    Vector::Vector(const Vector& input)
    {
        (*this) = input;
    }
    Vector::Vector(const Matrix& input, Index col)
    {
        Resize(input.GetRows());

        for (Index i = 0; i < numberRows_; ++i)
        {
            data_[i] = input(i, col);
        }
    }
    Vector::Vector(NumberRows numberRows)
    {
        Resize(numberRows);
    }
    Vector::Vector(NumberRows numberRows, Scalar value)
    {
        Resize(numberRows, value);
    }
    void Vector::Resize(NumberRows numberRows)
    {
        numberRows_ = numberRows;
        
        try 
        {
            data_ = Scalars(numberRows_, 0.0);
        }
        catch (const std::bad_alloc& e)
        {
            logger::Error(headerEilig, "Exception allocate memory for vector: %s", e.what());
            logger::Error(headerEilig, "Could not allocate memory for vector: %zu (kb)", size_t(sizeof(Scalar) * numberRows_ / 1024));
            throw;
		}
    }
    void Vector::Resize(NumberRows numberRows, Scalar value)
    {
        Resize(numberRows);
        (*this) = value;
    }
    void Vector::Fill(Scalar value)
    {
        (*this) = value;
    }
    Scalar Vector::operator()(Index i) const
    {
        return data_[i];
    }
    Scalar& Vector::operator()(Index i)
    {
        return data_[i];
    }
    Vector& Vector::operator=(Scalar rhs)
    {
        for (Index i = 0; i < numberRows_; ++i)
        {
            data_[i] = rhs;
        }

        return *this;
    }
    Vector& Vector::operator=(Vector&& rhs) noexcept
    {
        if (&rhs == this)
        {
            return *this;
        }

        numberRows_ = rhs.numberRows_;
        data_ = Scalars(std::move(rhs.data_));

        return *this;
    }
    Vector& Vector::operator=(const Vector& rhs)
    {
        numberRows_ = rhs.numberRows_;
        data_ = rhs.data_;

        return *this;
    }
    Vector Vector::operator+(Scalar rhs) const
    {
        Vector res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res.data_[i] += rhs;
        }

        return res;
    }
    Vector Vector::operator+(const Vector& rhs) const
    {
        Vector res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res.data_[i] += rhs.data_[i];
        }

        return res;
    }
    Vector Vector::operator+() const
    {
        return (*this);
    }
    Vector operator+(Scalar lhs, const Vector& rhs)
    {
        return rhs + lhs;
    }
    Vector Vector::operator-(Scalar rhs) const
    {
        Vector res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res.data_[i] -= rhs;
        }

        return res;
    }
    Vector Vector::operator-(const Vector& rhs) const
    {
        Vector res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res.data_[i] -= rhs.data_[i];
        }

        return res;
    }
    Vector Vector::operator-() const
    {
        return -1.0 * (*this);
    }
    Vector operator-(Scalar lhs, const Vector& rhs)
    {
        return -rhs + lhs;
    }
    Vector Vector::operator*(Scalar rhs) const
    {
        Vector res(*this);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res.data_[i] *= rhs;
        }

        return res;
    }
    Vector operator*(Scalar lhs, const Vector& rhs)
    {
        return rhs * lhs;
    }
    Vector& Vector::SwapRows(Index row1, Index row2)
    {
        Scalar temp;

        temp = data_[row1];
        data_[row2] = data_[row1];
        data_[row1] = temp;

        return *this;
    }
    Vector Vector::Region(Index row1, Index row2)
    {
        Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
        Index aux2 = row1 <= row2 ? row1 : row2;
        Vector res(aux1);

        for (Index i = 0; i < aux1; ++i)
        {
            res(i) = (*this)(aux2 + i);
        }

        return res;
    }
    void Vector::Region(Index row1, Index row2, const Vector& in)
    {
        Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
        Index aux2 = row1 <= row2 ? row1 : row2;

        for (Index i = 0; i < aux1; ++i)
        {
            (*this)(i + aux2) = in(i);
        }
    }
    NumberRows Vector::GetRows() const
    {
        return numberRows_;
    }
    NumberCols Vector::GetCols() const
    {
        return 1;
    }
    Scalar Vector::GetValue(Index i) const
    {
        return data_[i];
    }
    const Scalars& Vector::GetData() const
    {
        return data_;
    }
    void Vector::SetValue(Index i, Scalar value)
    {
        (*this)(i) = value;
    }
} /* namespace eilig */