#ifndef EILIG_VECTOR_HPP_
#define EILIG_VECTOR_HPP_

#include "eilig_types.hpp"
#include "eilig_matrix.hpp"

namespace eilig
{
    class Vector
    {
    public:
        Vector();
        Vector(const Vector& input);
        Vector(const Matrix& input, Index col);
        Vector(NumberRows numberRows);
        Vector(NumberRows numberRows, Scalar value);
        Vector(Vector&& input) noexcept;

        ~Vector() = default;

        void Resize(NumberRows numberRows);
        void Resize(NumberRows numberRows, Scalar value);
        void Fill(Scalar value);

        Scalar operator()(Index row) const;
        Scalar& operator()(Index row);

        Vector& operator=(Scalar rhs);
        Vector& operator=(const Vector& rhs);
        Vector& operator=(Vector&& rhs) noexcept;

        Vector operator+(Scalar rhs) const;
        Vector operator+(const Vector& rhs) const;
        Vector operator+() const;
        friend Vector operator+(Scalar lhs, const Vector& rhs);

        Vector operator-(Scalar rhs) const;
        Vector operator-(const Vector& rhs) const;
        Vector operator-() const;
        friend Vector operator-(Scalar lhs, const Vector& rhs);

        Vector operator*(Scalar rhs) const;
        friend Vector operator*(Scalar lhs, const Vector& rhs);

        Vector& SwapRows(Index row1, Index row2);
        Vector  Region(Index row1, Index row2);
        void    Region(Index row1, Index row2, const Vector& in);

        NumberRows GetRows() const;
        NumberCols GetCols() const;
        Scalar GetValue(Index row) const;
        const Scalars& GetData() const;

        void SetValue(Index row, Scalar value);

        friend void Add(Vector& out, const Vector& in, Scalar value);
        friend void Add(Vector& out, const Vector& in, const Vector& value);
        friend void Sub(Vector& out, const Vector& in, Scalar value);
        friend void Sub(Vector& out, const Vector& in, const Vector& value);
        friend void Mul(Vector& out, const Vector& in, Scalar value);

    private:
        NumberRows numberRows_{ 0 };

        Scalars data_{};
    };

} /* namespace eilig */

#endif /* EILIG_VECTOR_HPP_ */