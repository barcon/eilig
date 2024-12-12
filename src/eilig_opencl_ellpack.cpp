#include "eilig_matrix_ellpack_gpu.hpp"

namespace eilig
{
	namespace gpu
	{
        Ellpack::Ellpack()
        {
            Resize(1, 1);
        }
        Ellpack::Ellpack(const Ellpack& input)
        {
            *this = input;
        }
        Ellpack::Ellpack(const Matrix& input)
        {
            (*this) = input;
        }
        Ellpack::Ellpack(std::size_t rows, std::size_t cols)
        {
            Resize(rows, cols);
        }
        Ellpack::Ellpack(std::size_t rows, std::size_t cols, Scalar value) : Ellpack(rows, cols)
        {
            Fill(value);
        }
        /*
        bool Ellpack::IsUsed(std::size_t i, std::size_t j, std::size_t& position) const
        {
            for (std::size_t k = 0; k < count_[i]; ++k)
            {
                if (j == position_[i * width_ + k])
                {
                    position = k;
                    return true;
                }
                else if (j < position_[i * width_ + k])
                {
                    position = k;
                    return false;
                }
            }

            position = count_[i];
            return false;
        }
        std::size_t Ellpack::Add(std::size_t i, std::size_t j)
        {
            std::size_t position;
            std::size_t index;

            if (IsUsed(i, j, position))
            {
                index = i * width_ + position;
            }
            else
            {
                if (count_[i] == width_)
                {
                    Expand();
                }

                ShiftRight(i, position);
                index = i * width_ + position;
                data_[index] = 0.;
                position_[index] = j;
                count_[i] += 1;
            }

            return index;
        }
        void Ellpack::Remove(std::size_t i, std::size_t j)
        {
            std::size_t position;

            if (IsUsed(i, j, position))
            {
                ShiftLeft(i, position);
                count_[i] -= 1;
                data_[i * width_ + count_[i]] = 0;
                position_[i * width_ + count_[i]] = 0;

                Shrink();
            }
        }
        void Ellpack::Expand()
        {
            std::size_t delta = GrowthRate();
            std::size_t expansion = (width_ + delta) < cols_ ? width_ + delta : cols_;

            if (expansion > width_)
            {
                Array data(rows_ * expansion);
                Indices position(rows_ * expansion);

                for (std::size_t i = 0; i < rows_; ++i)
                {
                    for (std::size_t j = 0; j < width_; ++j)
                    {
                        data[i * expansion + j] = data_[i * width_ + j];
                        position[i * expansion + j] = position_[i * width_ + j];
                    }
                }

                data_ = std::move(data);
                position_ = std::move(position);
                width_ = data_.size() / rows_;
            }
        }
        void Ellpack::Shrink()
        {
            std::size_t minimum = *std::max_element(count_.begin(), count_.end());
            std::size_t delta = GrowthRate();
            std::size_t shrinkage = std::max(delta, minimum);

            if (shrinkage < width_)
            {
                Array data(rows_ * shrinkage);
                Indices position(rows_ * shrinkage);

                for (std::size_t i = 0; i < rows_; ++i)
                {
                    for (std::size_t j = 0; j < width_; ++j)
                    {
                        data[i * shrinkage + j] = data_[i * width_ + j];
                        position[i * shrinkage + j] = position_[i * width_ + j];
                    }
                }

                data_ = std::move(data);
                position_ = std::move(position);
                width_ = data_.size() / rows_;
            }
        }
        std::size_t Ellpack::GrowthRate()
        {
            return static_cast<std::size_t>(std::max(5., std::ceil(0.05 * cols_)));
        }
        void Ellpack::ShiftRight(std::size_t i, std::size_t position)
        {
            for (std::size_t k = count_[i]; k > position; --k)
            {
                data_[i * width_ + k] = data_[i * width_ + k - 1];
                position_[i * width_ + k] = position_[i * width_ + k - 1];
            }
        }
        void Ellpack::ShiftLeft(std::size_t i, std::size_t position)
        {
            for (std::size_t k = position; k < count_[i]; ++k)
            {
                data_[i * width_ + k] = data_[i * width_ + k + 1];
                position_[i * width_ + k] = position_[i * width_ + k + 1];
            }
        }
        void Ellpack::Resize(std::size_t rows, std::size_t cols)
        {
            if (rows == 0 || cols == 0)
            {
                return;
            }

            rows_ = rows;
            cols_ = cols;
            count_.resize(rows_);
            width_ = 0;

            for (auto& it : count_)
            {
                it = 0;
            }

            Expand();
        }
        void Ellpack::Fill(Scalar value)
        {
            (*this) = value;
        }
        Scalar Ellpack::operator()(std::size_t i, std::size_t j) const
        {
            Scalar res{ 0. };
            std::size_t position;

            if (IsUsed(i, j, position))
            {
                return data_[i * width_ + position];
            }

            return res;
        }
        Scalar& Ellpack::operator()(std::size_t i, std::size_t j)
        {
            std::size_t position;

            if (IsUsed(i, j, position))
            {
                return data_[i * width_ + position];
            }
            else
            {
                position = Add(i, j);
                return data_[position];
            }
        }
        Ellpack& Ellpack::operator=(Scalar rhs)
        {
            for (std::size_t i = 0; i < rows_; ++i)
            {
                for (std::size_t j = 0; j < cols_; ++j)
                {
                    (*this) = rhs;
                }
            }

            return *this;
        }
        Ellpack& Ellpack::operator=(const Ellpack& rhs)
        {
            rows_ = rhs.rows_;
            cols_ = rhs.cols_;
            width_ = rhs.width_;
            data_ = rhs.data_;
            position_ = rhs.position_;
            count_ = rhs.count_;

            return *this;
        }
        Ellpack& Ellpack::operator=(const Matrix& rhs)
        {
            std::size_t rows{ rhs.GetRows() };
            std::size_t cols{ rhs.GetCols() };

            Resize(rows, cols);

            for (std::size_t i = 0; i < rows; ++i)
            {
                for (std::size_t j = 0; j < cols; ++j)
                {
                    if (rhs(i, j) != 0.0)
                    {
                        (*this)(i, j) = rhs(i, j);
                    }
                }
            }


            return *this;
        }
        Ellpack& Ellpack::operator=(Ellpack&& rhs)
        {
            rows_ = std::move(rhs.rows_);
            cols_ = std::move(rhs.cols_);
            width_ = std::move(rhs.width_);
            data_ = std::move(rhs.data_);
            position_ = std::move(rhs.position_);
            count_ = std::move(rhs.count_);

            return *this;
        }

        Ellpack Ellpack::operator+(Scalar rhs) const
        {
            Ellpack res(*this);

            if (rhs == 0.)
            {
                return res;
            }

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.cols_; ++j)
                {
                    res(i, j) += rhs;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator+(const Ellpack& rhs) const
        {
            Ellpack res(*this);

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < rhs.count_[i]; ++j)
                {
                    auto col = rhs.position_[j];
                    auto value = rhs.data_[j];

                    res(i, col) += value;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator+() const
        {
            Ellpack res(*this);

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.count_[i]; ++j)
                {
                    res.data_[i * res.width_ + j] = +res.data_[i * res.width_ + j];
                }
            }
            return res;
        }
        Ellpack operator+(Scalar lhs, const Ellpack& rhs)
        {
            Ellpack res(rhs);

            if (lhs == 0.)
            {
                return res;
            }

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.cols_; ++j)
                {
                    res(i, j) += lhs;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator-(Scalar rhs) const
        {
            Ellpack res(*this);

            if (rhs == 0.)
            {
                return res;
            }

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.cols_; ++j)
                {
                    res(i, j) -= rhs;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator-(const Ellpack& rhs) const
        {
            Ellpack res(*this);

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < rhs.count_[i]; ++j)
                {
                    auto col = rhs.position_[j];
                    auto value = rhs.data_[j];

                    res(i, col) -= value;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator-() const
        {
            Ellpack res(*this);

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.count_[i]; ++j)
                {
                    res.data_[i * res.width_ + j] = -res.data_[i * res.width_ + j];
                }
            }
            return res;
        }
        Ellpack operator-(Scalar lhs, const Ellpack& rhs)
        {
            Ellpack res(rhs);

            if (lhs == 0.)
            {
                return res;
            }

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.cols_; ++j)
                {
                    res(i, j) = lhs - res(i, j);
                }
            }

            return res;
        }
        Ellpack Ellpack::operator*(Scalar rhs) const
        {
            Ellpack res(*this);

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.count_[i]; ++j)
                {
                    res.data_[i * res.width_ + j] = res.data_[i * res.width_ + j] * rhs;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator*(const Ellpack& rhs) const
        {
            Ellpack res(rows_, rhs.cols_);

            for (std::size_t i = 0; i < rows_; ++i)
            {
                for (std::size_t j = 0; j < rhs.cols_; ++j)
                {
                    Scalar aux1 = 0.;

                    for (std::size_t k = 0; k < count_[i]; ++k)
                    {
                        aux1 += data_[i * width_ + k] * rhs(position_[i * width_ + k], j);
                    }

                    res(i, j) = aux1;
                }
            }

            return res;
        }
        Vector Ellpack::operator*(const Vector& rhs) const
        {
            Vector res(rows_);

            for (std::size_t i = 0; i < rows_; ++i)
            {
                res(i) = 0.0;

                for (std::size_t k = 0; k < count_[i]; ++k)
                {
                    res(i) += data_[i * width_ + k] * rhs(position_[i * width_ + k]);
                }
            }

            return res;
        }
        Ellpack operator*(Scalar lhs, const Ellpack& rhs)
        {
            Ellpack res(rhs);

            for (std::size_t i = 0; i < res.rows_; ++i)
            {
                for (std::size_t j = 0; j < res.count_[i]; ++j)
                {
                    res.data_[i * res.width_ + j] *= lhs;
                }
            }

            return res;
        }
        std::ostream& operator<<(std::ostream& stream, const Ellpack& matrix)
        {
            for (std::size_t i = 0; i < matrix.GetRows(); ++i)
            {
                for (std::size_t j = 0; j < matrix.GetCols(); ++j)
                {
                    stream << utils::string::Format("%12.4g", matrix(i, j));
                }
                stream << "\n";
            }

            return stream;
        }
        Ellpack& Ellpack::SwapRows(std::size_t row1, std::size_t row2)
        {
            Scalar dataT;
            std::size_t positionT;
            std::size_t countT;

            for (std::size_t j = 0; j < width_; ++j)
            {
                dataT = data_[row1 * width_ + j];
                positionT = position_[row1 * width_ + j];

                data_[row1 * width_ + j] = data_[row2 * width_ + j];
                position_[row1 * width_ + j] = position_[row2 * width_ + j];

                data_[row2 * width_ + j] = dataT;
                position_[row2 * width_ + j] = positionT;

                countT = count_[row1];
                count_[row1] = count_[row2];
                count_[row2] = countT;
            }
            return *this;
        }
        Ellpack& Ellpack::SwapCols(std::size_t col1, std::size_t col2)
        {
            Scalar temp;

            for (std::size_t i = 0; i < rows_; ++i)
            {
                temp = (*this)(i, col1);
                (*this)(i, col1) = (*this)(i, col2);
                (*this)(i, col2) = temp;
            }

            return *this;
        }
        Ellpack Ellpack::Transpose() const
        {
            Ellpack res(cols_, rows_);

            for (std::size_t i = 0; i < rows_; ++i)
            {
                for (std::size_t j = 0; j < count_[i]; ++j)
                {
                    res(position_[i * width_ + j], i) = (*this)(i, j);
                }
            }

            return res;
        }
        Ellpack Ellpack::Diagonal() const
        {
            Ellpack res(rows_, cols_);

            for (std::size_t i = 0; (i < rows_) && (i < cols_); ++i)
            {
                res(i, i) = (*this)(i, i);
            }

            return res;
        }
        Ellpack Ellpack::Lower(bool diag) const
        {
            Ellpack res(rows_, cols_);

            if (diag)
            {
                for (std::size_t i = 0; i < rows_; ++i)
                {
                    for (std::size_t j = 0; j < count_[i]; ++j)
                    {
                        auto col = position_[i * width_ + j];

                        if (col <= i)
                        {
                            res(i, col) = data_[i * width_ + j];
                        }
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < rows_; ++i)
                {
                    for (std::size_t j = 0; j < count_[i]; ++j)
                    {
                        auto col = position_[i * width_ + j];

                        if (col < i)
                        {
                            res(i, col) = data_[i * width_ + j];
                        }
                    }
                }
            }

            return res;
        }
        Ellpack Ellpack::Upper(bool diag) const
        {
            Ellpack res(rows_, cols_);

            if (diag)
            {
                for (std::size_t i = 0; i < rows_; ++i)
                {
                    for (std::size_t j = 0; j < count_[i]; ++j)
                    {
                        auto col = position_[i * width_ + j];

                        if (col >= i)
                        {
                            res(i, col) = data_[i * width_ + j];
                        }
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < rows_; ++i)
                {
                    for (std::size_t j = 0; j < count_[i]; ++j)
                    {
                        auto col = position_[i * width_ + j];

                        if (col > i)
                        {
                            res(i, col) = data_[i * width_ + j];
                        }
                    }
                }
            }

            return res;
        }
        Ellpack Ellpack::Region(std::size_t row1, std::size_t col1, std::size_t row2, std::size_t col2)
        {
            std::size_t aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            std::size_t aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
            std::size_t aux3;
            std::size_t aux4;
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

            for (unsigned int i = 0; i < aux1; ++i)
            {
                for (unsigned int j = 0; j < count_[aux3 + i]; ++j)
                {
                    auto col = position_[(aux3 + i) * width_ + j];

                    if ((col >= aux4) && (col <= (aux4 + aux2)))
                    {
                        res(i, col - aux4) = data_[(aux3 + i) * width_ + j];
                    }
                }
            }

            return res;
        }
        void   Ellpack::Region(std::size_t row1, std::size_t col1, std::size_t row2, std::size_t col2, const Ellpack& in)
        {
            std::size_t aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            std::size_t aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
            std::size_t aux3;
            std::size_t aux4;

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

            for (std::size_t i = 0; i < aux1; ++i)
            {
                for (std::size_t j = 0; j < aux2; ++j)
                {
                    (*this)(aux3 + i, aux4 + j) = in(i, j);
                }
            }
        }

        std::size_t Ellpack::GetRows() const
        {
            return rows_;
        }
        std::size_t Ellpack::GetCols() const
        {
            return cols_;
        }

        void Add(Ellpack& out, const Ellpack& in, Scalar value)
        {
            out = in + value;
        }
        void Add(Ellpack& out, const Ellpack& in, const Ellpack& value)
        {
            out = in + value;
        }
        void Sub(Ellpack& out, const Ellpack& in, Scalar value)
        {
            out = in - value;
        }
        void Sub(Ellpack& out, const Ellpack& in, const Ellpack& value)
        {
            out = in - value;
        }
        void Mul(Ellpack& out, const Ellpack& in, Scalar value)
        {
            out = in * value;
        }
        void Mul(Ellpack& out, const Ellpack& in, const Ellpack& value)
        {
            out = in * value;
        }
        void Mul(Vector& out, const Ellpack& in, const Vector& value)
        {
            out = in * value;
        }
        */
	}
} /* namespace eilig */