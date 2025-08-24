#ifndef EILIG_OPENCL_VECTOR_HPP_
#define EILIG_OPENCL_VECTOR_HPP_

#include "eilig_types.hpp"
#include "eilig_vector.hpp"

#include "eilig_opencl_kernels.hpp"
#include "eilig_opencl_entry_proxy.hpp"

namespace eilig
{
    namespace opencl
    {
        class Vector
        {
        public:
            Vector(KernelsPtr kernels);
            Vector(KernelsPtr kernels, const Scalars& values);
            Vector(const Vector& input);
            Vector(KernelsPtr kernels, const eilig::Vector& input);
            Vector(KernelsPtr kernels, NumberRows numberRows);
            Vector(KernelsPtr kernels, NumberRows numberRows, Scalar value);
            Vector(Vector&& input) noexcept;

            ~Vector() = default;

            void Resize(NumberRows numberRows);
            void Resize(NumberRows numberRows, Scalar value);
            void Fill(Scalar value);

            EntryProxy operator()(Index row);

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

            friend std::ostream& operator<< (std::ostream& stream, const Vector& vector);

            Vector& SwapRows(Index row1, Index row2);
            Vector  Region(Index row1, Index row2);
            void    Region(Index row1, Index row2, const Vector& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            Scalar GetValue(Index row) const;
            KernelsPtr GetKernels() const;
            BufferPtr GetDataGPU () const;

            void SetValue(Index row, Scalar value);

        private:
            NumberRows numberRows_{ 0 };
            
            KernelsPtr kernels_{ nullptr };
            BufferPtr dataGPU_{ nullptr };
        };
    }

} /* namespace eilig */

#endif /* EILIG_OPENCL_VECTOR_HPP_ */