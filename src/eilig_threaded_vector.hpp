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
            Vector(const Devices& devices);
            Vector(const Devices& devices, const std::initializer_list<Scalar>& values);
            Vector(const Vector& input);
            Vector(const Devices& devices, const eilig::Vector& input);
            Vector(const Devices& devices, NumberRows numberRows);
            Vector(const Devices& devices, NumberRows numberRows, Scalar value);
            Vector(Vector&& input) noexcept;
            
            ~Vector() = default;

            void Resize(NumberRows numberRows);
            void Resize(NumberRows numberRows, Scalar value);
            void Fill(Scalar value);

            EntryProxyVector operator()(Index row);
            
            Vector& operator=(Scalar rhs);
            Vector& operator=(const Vector& rhs);
            Vector& operator=(Vector&& rhs) noexcept;

            Vector operator+(Scalar rhs) const;
            //Vector operator+(const Vector& rhs) const;
            //Vector operator+() const;
            //friend Vector operator+(Scalar lhs, const Vector& rhs);

            //Vector operator-(Scalar rhs) const;
            //Vector operator-(const Vector& rhs) const;
            //Vector operator-() const;
            //friend Vector operator-(Scalar lhs, const Vector& rhs);

            //Vector operator*(Scalar rhs) const;
            //friend Vector operator*(Scalar lhs, const Vector& rhs);

            //Vector& SwapRows(Index row1, Index row2);
            //Vector  Region(Index row1, Index row2) const;
            //void    Region(Index row1, Index row2, const Vector& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            Scalar GetValue(Index row) const;

            //void SetValue(Index row, Scalar value);

        private:
			void SetDevices(const Devices& devices);

            Devices devices_;
            NumberRows numberRows_{ 0 };    
        };
    }
} /* namespace eilig */

#endif /* EILIG_THREADED_VECTOR_HPP_ */