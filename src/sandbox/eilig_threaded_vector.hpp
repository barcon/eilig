#ifndef EILIG_THREADED_VECTOR_HPP_
#define EILIG_THREADED_VECTOR_HPP_

#include "eilig_threaded.hpp"
#include "eilig_threaded_entry_proxy.hpp"

namespace eilig
{
    namespace threaded
    {
        class Vector
        {
        public:
            Vector(const Vector& input);
            Vector(const Devices& devices);
            Vector(const Devices& devices, const std::initializer_list<Scalar>& value);
            Vector(const Devices& devices, const eilig::Vector& input);
            Vector(const Devices& devices, NumberRows numberRows);
            Vector(const Devices& devices, NumberRows numberRows, Scalar value);
            Vector(Vector&& input) noexcept;
            
            ~Vector() = default;

            eilig::Vector Convert() const;

            void Resize(NumberRows numberRows);
            void Resize(NumberRows numberRows, Scalar value);
            void Fill(Scalar value);

            EntryProxyVector operator()(Index row);
            
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
            Vector  Region(Index row1, Index row2) const;
            void    Replace(Index row1, const Vector& in);
            void    Replace(Index row1, const eilig::Vector& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            Scalar GetValue(Index row) const;

            void Equal(Index row, Scalar value);
            void Equal(Scalar value);
            void Equal(const Vector& value);
            void Equal(const eilig::Vector& value);
            void Equal(const std::initializer_list<Scalar>& value);

            void Add(Scalar value);
            void Add(const Vector& value);
            void Sub(Scalar value);
            void Sub(const Vector& value);
            void Mul(Scalar value);

			const Blocks& GetBlocks() const;

			friend threaded::Ellpack;

        private:
			void SetDevices(const Devices& devices);

            Blocks blocks_;

            NumberRows numberRows_{ 0 }; 
        };
    }
} /* namespace eilig */

#endif /* EILIG_THREADED_VECTOR_HPP_ */