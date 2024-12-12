#ifndef EILIG_MATRIX_ELLPACK_GPU_HPP_
#define EILIG_MATRIX_ELLPACK_GPU_HPP_

#include "eilig_types.hpp"
#include "eilig_vector.hpp"
#include "eilig_matrix.hpp"

namespace eilig
{
    namespace opencl
    {
        class Ellpack
        {
        public:
            Ellpack();
            Ellpack(const Ellpack& input);
            Ellpack(const Matrix& input);
            Ellpack(std::size_t rows, std::size_t cols);
            Ellpack(std::size_t rows, std::size_t cols, Scalar value);

            ~Ellpack() = default;

            //bool IsUsed(std::size_t i, std::size_t j, std::size_t& position) const;
            //std::size_t Add(std::size_t i, std::size_t j);
            //void Remove(std::size_t i, std::size_t j);

            //void Resize(std::size_t rows, std::size_t cols);
            //void Fill(Scalar value);

            //inline Scalar operator()(std::size_t i, std::size_t j) const;
            //inline Scalar& operator()(std::size_t i, std::size_t j);

            //Ellpack& operator=(Scalar rhs);
            //Ellpack& operator=(const Ellpack& rhs);
            //Ellpack& operator=(const Matrix& rhs);
            //Ellpack& operator=(Ellpack&& rhs);

            //Ellpack operator+(Scalar rhs) const;
            //Ellpack operator+(const Ellpack& rhs) const;
            //Ellpack operator+() const;
            //friend Ellpack operator+(Scalar lhs, const Ellpack& rhs);

            //Ellpack operator-(Scalar rhs) const;
            //Ellpack operator-(const Ellpack& rhs) const;
            //Ellpack operator-() const;
            //friend Ellpack operator-(Scalar lhs, const Ellpack& rhs);

            //Ellpack operator*(Scalar rhs) const;
            //Ellpack operator*(const Ellpack& rhs) const;
            //Vector operator*(const Vector& rhs) const;
            //friend Ellpack operator*(Scalar lhs, const Ellpack& rhs);

            //friend std::ostream& operator<< (std::ostream& stream, const Ellpack& Ellpack);

            //Ellpack& SwapRows(std::size_t row1, std::size_t row2);
            //Ellpack& SwapCols(std::size_t cols1, std::size_t cols2);        
            //Ellpack Transpose() const;
            //Ellpack Diagonal() const;
            //Ellpack Lower(bool diag) const;
            //Ellpack Upper(bool diag) const;
            //Ellpack Region(std::size_t row1, std::size_t col1, std::size_t row2, std::size_t col2);
            //void   Region(std::size_t row1, std::size_t col1, std::size_t row2, std::size_t col2, const Ellpack& in);

            //std::size_t GetRows() const;
            //std::size_t GetCols() const;

        private:
            //void Expand();
            //void Shrink();
            //std::size_t GrowthRate();
            //void ShiftRight(std::size_t i, std::size_t position);
            //void ShiftLeft(std::size_t i, std::size_t position);

            std::size_t rows_{ 0 };
            std::size_t cols_{ 0 };
            std::size_t width_{ 0 };

            Array data_{};
            Indices position_{};
            Indices count_{};
        };

        //void Add(Ellpack& out, const Ellpack& in, Scalar value);
        //void Add(Ellpack& out, const Ellpack& in, const Ellpack& value);
        //void Sub(Ellpack& out, const Ellpack& in, Scalar value);
        //void Sub(Ellpack& out, const Ellpack& in, const Ellpack& value);
        //void Mul(Ellpack& out, const Ellpack& in, Scalar value);
        //void Mul(Ellpack& out, const Ellpack& in, const Ellpack& value);
        //void Mul(Vector& out, const Ellpack& in, const Vector& value);
    }
} /* namespace eilig */

#endif /* EILIG_MATRIX_ELLPACK_GPU_HPP_ */