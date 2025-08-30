#include "eilig_matrix_ellpack.hpp"

namespace eilig
{
    Ellpack::Ellpack()
    {
        Resize(1, 1);
    }
    Ellpack::Ellpack(Ellpack&& input) noexcept
    {
        (*this) = std::move(input);
    }
    Ellpack::Ellpack(const Ellpack& input)
    {
        (*this) = input;
    }
    Ellpack::Ellpack(const std::vector<Scalars>& values)
    {
        Resize(values.size(), values[0].size());

        for (Index i = 0; i < numberRows_; i++)
        {
            for (Index j = 0; j < numberCols_; j++)
            {
                if (utils::math::IsAlmostEqual(values[i][j], 0.0, 5))
                {
                    continue;
                }

                (*this)(i, j) = values[i][j];
            }
        }
    }
    Ellpack::Ellpack(const eilig::Matrix& input)
    {
        Resize(input.GetRows(), input.GetCols());

        for (Index i = 0; i < numberRows_; i++)
        {
            for (Index j = 0; j < numberCols_; j++)
            {             
                if (utils::math::IsAlmostEqual(input(i, j), 0.0, 5))
                {
                    continue;
                }

                (*this)(i, j) = input(i, j);
            }
        }
    }
    Ellpack::Ellpack(NumberRows numberRows, NumberCols numberCols)
    {
        Resize(numberRows, numberCols);
    }
    Ellpack::Ellpack(NumberRows numberRows, NumberCols numberCols, Type type)
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
    bool Ellpack::IsUsed(Index row, Index col) const
    {
        Index position{ 0 };

        return IsUsed(row, col, position);
    }
    bool Ellpack::IsUsed(Index row, Index col, Index& position) const
    {
        Index count;

        count = count_[row];

        for (Index k = 0; k < count; ++k)
        {
            if (col == position_[row * width_ + k])
            {
                position = k;
                return true;
            }
            else if (position_[row * width_ + k] > col)
            {
                position = k;
                return false;
            }
        }

        position = count;
        return false;
    }
    Index Ellpack::Add(Index row, Index col)
    {      
        Index count{ 0 };
        Index position{ 0 };
        Scalar zeroScalar{ 0.0 };

        count = count_[row];

        if (!IsUsed(row, col, position))
        {
            if (position < count)
            {
                if (count == width_)
                {
                    Expand(width_ + GrowthRate());
                }

                ShiftRight(row, position);
            }
            else if (position == count)
            {
                if (count == width_)
                {
                    Expand(width_ + GrowthRate());
                }
            }
                    
            count += 1;
            count_[row] = count;
            position_[row * width_ + position] = col;
            data_[row * width_ + position] = zeroScalar;
        }

        //std::cout << "Add Index = " << row * width_ + position << std::endl;
        return (row * width_ + position);
    }
    void Ellpack::Remove(Index row, Index col)
    {
        Index count{ 0 };
        Index position{ 0 };
        Index zeroIndex{ 0 };
        Scalar zeroScalar{ 0.0 };

        count = count_[row];

        if (IsUsed(row, col, position))
        {
            ShiftLeft(row, position);
            
            position_[row * width_ + count - 1] = zeroIndex;
            data_[row * width_ + count - 1] = zeroScalar;
            count -= 1;
            count_[row] = count;

            Shrink();
        }
    }
    void Ellpack::Expand(NumberCols width)
    {
        Index expansion = width == 0 ? GrowthRate(): width;

        expansion = expansion > numberCols_ ? numberCols_ : expansion;
 
        if (expansion > width_)
        {
            Scalars data(numberRows_ * expansion, 0.0);
            Indices position(numberRows_ * expansion, 0);

            for (Index i = 0; i < numberRows_; ++i)
            {
                for (Index j = 0; j < count_[i]; ++j)
                {
                    data[i * expansion + j] = data_[i * width_ + j];
                    position[i * expansion + j] = position_[i * width_ + j];
                }
            }

            position_ = Indices(std::move(position));
            data_= Scalars(std::move(data));
            width_ = expansion;
        }
    }
    void Ellpack::Shrink()
    {
        Index shrinkage = std::max({ MaxCount(), GrowthRate()});

        shrinkage = shrinkage > numberCols_ ? numberCols_ : shrinkage;

        if (shrinkage < width_)
        {
            Indices position(numberRows_ * shrinkage, 0);
            Scalars data(numberRows_ * shrinkage, 0.0);

            for (Index i = 0; i < numberRows_; ++i)
            {
                for (Index j = 0; j < count_[i]; ++j)
                {
                    position[i * shrinkage + j] = position_[i * width_ + j];
                    data[i * shrinkage + j] = data_[i * width_ + j];
                }
            }

            position_ = Indices(std::move(position));
            data_ = Scalars(std::move(data));
            width_ = shrinkage;
        }
    }
    Index Ellpack::GrowthRate()
    {
        return static_cast<Index>(std::max(5.0, std::ceil(0.05 * numberCols_)));
    }
    Index Ellpack::MaxCount()
    {
        return *std::max_element(count_.begin(), count_.end());
    }
    void Ellpack::ShiftRight(Index row, Index position)
    {
        Index count{ 0 };

        count = count_[row];

        if (position < count)
        {
            for (Index k = count_[row]; k > position; --k)
            {
                position_[row * width_ + k] = position_[row * width_ + k - 1];
                data_[row * width_ + k] = data_[row * width_ + k - 1];
            };
        }
    }
    void Ellpack::ShiftLeft(Index row, Index position)
    {
        Index count{ 0 };

        count = count_[row];

        if (position < count)
        {
            for (Index k = position; k < count_[row]; ++k)
            {
                position_[row * width_ + k] = position_[row * width_ + k + 1];
                data_[row * width_ + k] = data_[row * width_ + k + 1];
            }
        }
    }
    void Ellpack::Resize(NumberRows numberRows, NumberCols numberCols)
    {
        numberRows_ = numberRows;
        numberCols_ = numberCols;
        width_ = GrowthRate() > numberCols_ ? numberCols_ : GrowthRate();
        count_ = Indices(numberRows_, 0);
        position_ = Indices(numberRows_ * width_, 0);
        data_ = Scalars(numberRows_ * width_, 0.0);
    }
    void Ellpack::Resize(NumberRows numberRows, NumberCols numberCols, Scalar value)
    {
        numberRows_ = numberRows;
        numberCols_ = numberCols;
        width_ = GrowthRate() > numberCols_ ? numberCols_ : GrowthRate();
        count_ = Indices(numberRows_, 0);
        position_ = Indices(numberRows_ * width_, 0);
        data_ = Scalars(numberRows_ * width_, value);
    }
    void Ellpack::Fill(Scalar value)
    {
        (*this) = value;
    }
    void Ellpack::Clear()
    {
        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                position_[i * width_ + j] = 0;
                data_[i * width_ + j] = 0.0;
            }

            count_[i] = 0;
        }

    }
    void Ellpack::Dump() const
    {
        logger::Info(headerEilig, "Matrix Ellpack (%zu x %zu):", numberRows_, numberCols_);

        std::cout << "Rows: " << numberRows_ << std::endl;
        std::cout << "Cols: " << numberCols_ << std::endl;
        std::cout << "Width: " << width_ << std::endl;
        std::cout << std::endl;

        std::cout << "Count: " << std::endl;
        for (Index i = 0; i < numberRows_; ++i)
        {
            std::cout << count_[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Position: " << std::endl;
        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < width_; ++j)
            {
                std::cout << position_[i * width_ + j] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Data: " << std::endl;
        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < width_; ++j)
            {
                std::cout << utils::string::Format("%12.5g", data_[i * width_ + j]) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    Scalar Ellpack::operator()(Index row, Index col) const
    {
        Scalar res{ 0. };
        Index position;

        if (IsUsed(row, col, position))
        {
            return data_[row * width_ + position];
        }

        return res;
    }
    Scalar& Ellpack::operator()(Index row, Index col)
    {
        Index index;

        index = Add(row, col);      
        return data_[index];
    }
    Ellpack& Ellpack::operator=(Scalar rhs)
    {
        if (utils::math::IsAlmostEqual(rhs, 0.0, 5))
        {
            Clear();
            Shrink();
            return *this;
        }

        Expand(numberCols_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            count_[i] = numberCols_;
            for (Index j = 0; j < count_[i]; ++j)
            {
                data_[i * width_ + j] = rhs;
                position_[i * width_ + j] = j;
            }
        }

        return *this;
    }
    Ellpack& Ellpack::operator=(Ellpack&& rhs) noexcept
    {
        if (&rhs == this)
        {
            return *this;
        }

        numberRows_ = rhs.numberRows_;
        numberCols_ = rhs.numberCols_;
        width_ = rhs.width_;
        count_ = Indices(std::move(rhs.count_));
        position_ = Indices(std::move(rhs.position_));
        data_ = Scalars(std::move(rhs.data_));

        return *this;
    }
    Ellpack& Ellpack::operator=(const Ellpack& rhs)
    {
        numberRows_ = rhs.numberRows_;
        numberCols_ = rhs.numberCols_;
        width_ = rhs.width_;
        position_ = rhs.position_;
        data_ = rhs.data_;
        count_ = rhs.count_;

        return *this;
    }
    Ellpack Ellpack::operator+(Scalar rhs) const
    {
        Ellpack res;
        Index k;
        
        if (utils::math::IsAlmostEqual(rhs, 0.0, 5))
        {
            return *this;
        }

        res.Resize(numberRows_, numberCols_, rhs);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                k = position_[i * width_ + j];
                res.data_[i * res.width_ + k] += data_[i * width_ + j];
            }
        }

        return res;
    }
    Ellpack Ellpack::operator+(const Ellpack& rhs) const
    {
        Ellpack res(*this);

        for (Index i = 0; i < rhs.numberRows_; ++i)
        {          
            for (Index j = 0; j < rhs.count_[i]; ++j)
            {
                auto col = rhs.position_[i * rhs.width_ + j];
                auto value = rhs.data_[i * rhs.width_ + j];

                res(i, col) += value;
            }
        }

        return res;
    }
    Ellpack Ellpack::operator+() const
    {
        return (*this);
    }
    Ellpack operator+(Scalar lhs, const Ellpack& rhs)
    {
        return rhs + lhs;
    }
    Ellpack Ellpack::operator-(Scalar rhs) const
    {
        Ellpack res;
        Index k;

        if (utils::math::IsAlmostEqual(rhs, 0.0, 5))
        {
            return *this;
        }

        res.Resize(numberRows_, numberCols_, -rhs);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                k = position_[i * width_ + j];
                res.data_[i * res.width_ + k] += data_[i * width_ + j];
            }
        }

        return res;
    }
    Ellpack Ellpack::operator-(const Ellpack& rhs) const
    {
        Ellpack res(*this);

        for (Index i = 0; i < res.numberRows_; ++i)
        {
            for (Index j = 0; j < rhs.count_[i]; ++j)
            {
                auto col = rhs.position_[i * rhs.width_ + j];
                auto value = rhs.data_[i * rhs.width_ + j];

                res(i, col) -= value;
            }
        }

        return res;
    }
    Ellpack Ellpack::operator-() const
    {
        return -1.0 * (*this);
    }
    Ellpack operator-(Scalar lhs, const Ellpack& rhs)
    {
        return -rhs + lhs;
    }
    Ellpack Ellpack::operator*(Scalar rhs) const
    {
        Ellpack res(*this);

        for (Index i = 0; i < res.numberRows_; ++i)
        {
            for (Index j = 0; j < res.count_[i]; ++j)
            {
                res.data_[i * res.width_ + j] *= rhs;
            }
        }

        return res;
    }
    Ellpack Ellpack::operator*(const Ellpack& rhs) const
    {
        Ellpack res(numberRows_, rhs.numberCols_);
        Scalar sum;

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < rhs.numberCols_; ++j)
            {
                sum = 0.;

                for (Index k = 0; k < count_[i]; ++k)
                {
                    sum += data_[i * width_ + k] * rhs(position_[i * width_ + k], j);
                }

                if (sum != 0.0)
                {
                    res(i, j) = sum;
                }
            }
        }
        
        return res;
    }
    Vector Ellpack::operator*(const Vector& rhs) const
    {
        Vector res(numberRows_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            res(i) = 0.0;

            for (Index k = 0; k < count_[i]; ++k)
            {
                res(i) += data_[i * width_ + k] * rhs(position_[i * width_ + k]);
            }
        }

        return res;
    }
    Ellpack operator*(Scalar lhs, const Ellpack& rhs)
    {
        return rhs * lhs;
    }
    std::ostream& operator<<(std::ostream& stream, const Ellpack& matrix)
    {
        for (Index i = 0; i < matrix.GetRows(); ++i)
        {
            for (Index j = 0; j < matrix.GetCols(); ++j)
            {
                stream << utils::string::Format("%12.5g", matrix.GetValue(i, j));
            }
            stream << "\n";
        }

        return stream;
    }
    Ellpack& Ellpack::SwapRows(Index row1, Index row2)
    {
        Scalar dataT;
        Index positionT;
        Index countT;

        countT = count_[row1];
        count_[row1] = count_[row2];
        count_[row2] = countT;

        for (Index i = 0; i < width_; ++i)
        {
            positionT = position_[row1 * width_ + i];
            position_[row1 * width_ + i] = position_[row2 * width_ + i];
            position_[row2 * width_ + i] = positionT;

            dataT = data_[row1 * width_ + i];
            data_[row1 * width_ + i] = data_[row2 * width_ + i];
            data_[row2 * width_ + i] = dataT;
        }

        return *this;
    }
    Ellpack& Ellpack::SwapCols(Index col1, Index col2)
    {
        Scalar temp;

        for (Index i = 0; i < numberRows_; ++i)
        {
            temp = (*this)(i, col1);
            (*this)(i, col1) =  (*this)(i, col2);
            (*this)(i, col2) = temp;
        }

        return *this;
    }
    Ellpack Ellpack::Transpose() const
    {
        Ellpack res(numberCols_, numberRows_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                res(position_[i * width_ + j], i) = data_[i * width_ + j];
            }
        }

        return res;
    }
    Ellpack Ellpack::Diagonal() const
    {
        Ellpack res(numberRows_, numberCols_);

        for (Index i = 0; (i < numberRows_) && (i < numberCols_); ++i)
        {
            res(i, i) = (*this)(i, i);
        }

        return res;
    }
    Ellpack Ellpack::Lower(bool diag) const
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
    Ellpack Ellpack::LowerWithDiagonal() const
    {
        Ellpack res(numberRows_, numberCols_);

        res.Expand(width_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                auto col = position_[i * width_ + j];

                if (col <= i)
                {
                    res(i, col) = data_[i * width_ + j];
                }
            }
        }

        return res;
    }
    Ellpack Ellpack::LowerWithoutDiagonal() const
    {
        Ellpack res(numberRows_, numberCols_);

        res.Expand(width_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                auto col = position_[i * width_ + j];

                if (col < i)
                {
                    res(i, col) = data_[i * width_ + j];
                }
            }
        }

        return res;
    }
    Ellpack Ellpack::Upper(bool diag) const
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
    Ellpack Ellpack::UpperWithDiagonal() const
    {
        Ellpack res(numberRows_, numberCols_);
        
        res.Expand(width_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                auto col = position_[i * width_ + j];

                if (col >= i)
                {
                    res(i, col) = data_[i * width_ + j];
                }
            }
        }

        return res;
    }
    Ellpack Ellpack::UpperWithoutDiagonal() const
    {
        Ellpack res(numberRows_, numberCols_);

        res.Expand(width_);

        for (Index i = 0; i < numberRows_; ++i)
        {
            for (Index j = 0; j < count_[i]; ++j)
            {
                auto col = position_[i * width_ + j];

                if (col > i)
                {
                    res(i, col) = data_[i * width_ + j];
                }
            }
        }


        return res;
    }
    Ellpack Ellpack::Region(Index row1, Index col1, Index row2, Index col2) const
    {
        Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
        Index aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
        Index aux3;
        Index aux4;
        Ellpack res(aux1, aux2);


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

        for (unsigned int i = 0; i < aux1; i++)
        {
            for (unsigned int j = 0; j < count_[aux3 + i]; j++)
            {
                auto col = position_[(aux3 + i) * width_ + j];

                if ((col >= aux4) && (col < (aux4 + aux2)))
                {
                    res(i, col - aux4) = data_[(aux3 + i) * width_ + j];
                }
            }
        }

        return res;
    }
    void Ellpack::Region(Index row1, Index col1, Index row2, Index col2, const Ellpack& in)
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
                (*this)(aux3 + i, aux4 + j) = in.GetValue(i, j);
            }
        }
    }
    NumberRows Ellpack::GetRows() const
    {
        return numberRows_;
    }
    NumberCols Ellpack::GetCols() const
    {
        return numberCols_;
    }
    NumberCols Ellpack::GetWidth() const
    {
        return width_;
    }
    Scalar Ellpack::GetValue(Index row, Index col) const
    {
        Scalar res{ 0.0 };
        Index position;
        Index index;

        if (IsUsed(row, col, position))
        {
            index = row * width_ + position;

            res = data_[index];
        }

        return res;
    }
    const Indices& Ellpack::GetCount() const
    {
        return count_;
    }
    const Indices& Ellpack::GetPosition() const
    {
        return position_;
    }
    const Scalars& Ellpack::GetData() const
    {
        return data_;
    }
    void Ellpack::SetValue(Index i, Index j, Scalar value)
    {
        if (utils::math::IsAlmostEqual(value, 0.0, 5))
        {
            Remove(i, j);
        }
        else
        {
            (*this)(i, j) = value;
        }
    }
} /* namespace eilig */