#ifndef EILIG_OPENCL_VECTOR_HPP_
#define EILIG_OPENCL_VECTOR_HPP_

#include "eilig_types.hpp"
#include "eilig_vector.hpp"

#include "eilig_opencl_kernel.hpp"
#include "eilig_opencl_entry_proxy.hpp"

namespace eilig
{
    namespace opencl
    {
        class Vector
        {
        public:
            Vector();
            Vector(const Vector& input);
            Vector(const std::initializer_list<Scalar>& value);
            Vector(const eilig::Vector& input);
            Vector(NumberRows numberRows);
            Vector(NumberRows numberRows, Scalar value);
            Vector(Vector&& input) noexcept;

            ~Vector() = default;

            eilig::Vector Convert() const;

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

            Vector& SwapRows(Index row1, Index row2);
            Vector  Region(Index row1, Index row2) const;
            void    Replace(Index row1, const Vector& in);
            void    Replace(Index row1, const eilig::Vector& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            Scalar GetValue(Index row) const;
            BufferPtr GetDataGPU () const;
			KernelVectorPtr GetKernel() const;
			const DeviceIndex& GetDeviceIndex() const;

            void Equal(Index row, Scalar value);
            void Equal(Scalar value);
            void Equal(const Vector& value);
            void Equal(const eilig::Vector& value);
            void Equal(const std::initializer_list<Scalar>& value);

            void SetDevice(DeviceIndex deviceIndex);
			
        private:         
            void InitKernel();
            
            NumberRows numberRows_{ 0 };
			DeviceIndex deviceIndex_{ 0 };
            
            KernelVectorPtr kernel_{ nullptr };
            BufferPtr dataGPU_{ nullptr };
        };
    }

} /* namespace eilig */

#endif /* EILIG_OPENCL_VECTOR_HPP_ */