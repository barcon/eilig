#ifndef EILIG_THREADED_MATRIX_ELLPACK_HPP_
#define EILIG_THREADED_MATRIX_ELLPACK_HPP_

#include "eilig_threaded.hpp"
#include "eilig_threaded_entry_proxy.hpp"
#include "eilig_threaded_vector.hpp"

namespace eilig
{
    namespace threaded
    {
        class Ellpack
        {
        public:
            Ellpack(const Ellpack& input);
            Ellpack(const Devices& devices);
            Ellpack(const Devices& devices, const std::initializer_list<std::initializer_list<Scalar>>& value);
            Ellpack(const Devices& devices, const eilig::Ellpack& input);
            Ellpack(const Devices& devices, const eilig::Matrix& input);
            Ellpack(const Devices& devices, NumberRows numberRows, NumberCols numberCols);
            Ellpack(const Devices& devices, NumberRows numberRows, NumberCols numberCols, Type type);
            Ellpack(Ellpack&& input) noexcept;

            ~Ellpack() = default;

            eilig::Ellpack Convert() const;

            void Resize(NumberRows numberRows, NumberCols numberCols);
            void Resize(NumberRows numberRows, NumberCols numberCols, Scalar value);
            void Fill(Scalar value);

            EntryProxyEllpack operator()(Index row, Index col);

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
            //Ellpack operator*(const Ellpack& rhs) const;
            Vector operator*(const Vector& rhs) const;
            friend Ellpack operator*(Scalar lhs, const Ellpack& rhs);

            Ellpack& SwapRows(Index row1, Index row2);
            Ellpack& SwapCols(Index col1, Index col2);
            Scalar Trace() const;
            Scalar Sum() const;
            //Ellpack Transpose() const;
            Ellpack Diagonal() const;
            Ellpack DiagonalScale(Scalar factor) const;
            Vector  DiagonalVector() const;
            Ellpack Lower(bool diag) const;
            Ellpack LowerWithDiagonal() const;
            Ellpack LowerWithoutDiagonal() const;
            Ellpack Upper(bool diag) const;
            Ellpack UpperWithDiagonal() const;
            Ellpack UpperWithoutDiagonal() const;
            Ellpack Region(Index row1, Index col1, Index row2, Index col2) const;
            //void Replace(Index row1, Index col1, const Ellpack& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            Scalar GetValue(Index row, Index col) const;

            void Equal(Index row, Index col, Scalar value);
            void Equal(Scalar value);
            void Equal(const Ellpack& value);
            void Equal(const std::initializer_list<std::initializer_list<Scalar>>& value);

            void Add(Scalar value);
            void Add(const Ellpack& value);
            void Sub(Scalar value);
            void Sub(const Ellpack& value);
            void Mul(Scalar value);

            const Blocks& GetBlocks() const;

        private:
			void SetDevices(const Devices& devices);

            NumberRows numberRows_{ 0 };
            NumberCols numberCols_{ 0 };

			Blocks blocks_;
        };
    }
} /* namespace eilig */

#endif /* EILIG_THREADED_MATRIX_ELLPACK_HPP_ */