#ifndef EILIG_MATRIX_ELLPACK_HPP_
#define EILIG_MATRIX_ELLPACK_HPP_

#include "eilig_types.hpp"
#include "eilig_vector.hpp"
#include "eilig_matrix.hpp"

namespace eilig
{
    class Ellpack
    {
    public:
        Ellpack();
        Ellpack(const Ellpack& input);
        Ellpack(const eilig::Matrix& input);
        Ellpack(NumberRows numberRows, NumberCols numberCols);
        Ellpack(NumberRows numberRows, NumberCols numberCols, Type type);
        Ellpack(Ellpack&& input) noexcept;

        ~Ellpack() = default;

        bool IsUsed(Index row, Index col) const;
        bool IsUsed(Index row, Index col, Index& position) const;
        Index Add(Index row, Index col);
        void Remove(Index row, Index col);

        void Resize(NumberRows numberRows, NumberCols numberCols);
        void Resize(NumberRows numberRows, NumberCols numberCols, Scalar value);
        void Fill(Scalar value);
        void Dump() const;

        Scalar operator()(Index row, Index col) const;
        Scalar& operator()(Index row, Index col);

        Ellpack& operator=(Scalar rhs);
        Ellpack& operator=(const Ellpack& rhs);
        Ellpack& operator=(Ellpack&& rhs) noexcept;

        Ellpack operator+(Scalar rhs) const;
        Ellpack operator+(const Ellpack& rhs) const;
        Ellpack operator+() const;
        friend Ellpack operator+(Scalar lhs, const Ellpack& rhs);

        Ellpack operator-(Scalar rhs) const;
        Ellpack operator-(const Ellpack& rhs) const;
        Ellpack operator-() const;
        friend Ellpack operator-(Scalar lhs, const Ellpack& rhs);

        Ellpack operator*(Scalar rhs) const;
        Ellpack operator*(const Ellpack& rhs) const;
        Vector operator*(const Vector& rhs) const;
        friend Ellpack operator*(Scalar lhs, const Ellpack& rhs);

        Ellpack& SwapRows(Index row1, Index row2);
        Ellpack& SwapCols(Index col1, Index col2);
        Ellpack Transpose() const;
        Ellpack Diagonal() const;
        Ellpack Lower(bool diag) const;
        Ellpack LowerWithDiagonal() const;
        Ellpack LowerWithoutDiagonal() const;
        Ellpack Upper(bool diag) const;
        Ellpack UpperWithDiagonal() const;
        Ellpack UpperWithoutDiagonal() const;
        Ellpack Region(Index row1, Index col1, Index row2, Index col2) const;
        void   Region(Index row1, Index col1, Index row2, Index col2, const Ellpack& in);

        NumberRows GetRows() const;
        NumberCols GetCols() const;
        NumberCols GetWidth() const;
        Scalar GetValue(Index row, Index col) const;
        const Indices& GetCount() const;
        const Indices& GetPosition() const;
        const Scalars& GetData() const;

        void SetValue(Index row, Index col, Scalar value);

        //friend void Add(Ellpack& out, const Ellpack& in, Scalar value);
        //friend void Add(Ellpack& out, const Ellpack& in, const Ellpack& value);
        //friend void Sub(Ellpack& out, const Ellpack& in, Scalar value);
        //friend void Sub(Ellpack& out, const Ellpack& in, const Ellpack& value);
        //friend void Mul(Ellpack& out, const Ellpack& in, Scalar value);
        //friend void Mul(Ellpack& out, const Ellpack& in, const Ellpack& value);
        //friend void Mul(Vector& out, const Ellpack& in, const Vector& value);

#ifdef EILIG_ENABLE_OPENCL
        friend opencl::Ellpack;
#endif

    private:

        void Expand(NumberCols width);
        void Shrink();
        Index GrowthRate();
        Index MaxCount();
        void ShiftRight(Index row, Index position);
        void ShiftLeft(Index row, Index position);
        void Clear();

        NumberRows numberRows_{ 0 };
        NumberCols numberCols_{ 0 };
        NumberCols width_{ 0 };
        
        Indices count_{};
        Indices position_{};
        Scalars data_{};
    };
} /* namespace eilig */

#endif /* EILIG_MATRIX_EllPACK_HPP_ */