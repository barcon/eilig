#ifndef EILIG_OPENCL_MATRIX_ELLPACK_HPP_
#define EILIG_OPENCL_MATRIX_ELLPACK_HPP_

#include "eilig_types.hpp"
#include "eilig_matrix_ellpack.hpp"

#include "eilig_opencl_kernel.hpp"
#include "eilig_opencl_entry_proxy.hpp"
#include "eilig_opencl_vector.hpp"

namespace eilig
{
    namespace opencl
    {
        class Ellpack
        {
        public:
            Ellpack();
            Ellpack(const Ellpack& input);
            Ellpack(const std::initializer_list<std::initializer_list<Scalar>>& value);
            Ellpack(const eilig::Ellpack& input);
            Ellpack(const eilig::Matrix& input);
            Ellpack(NumberRows numberRows, NumberCols numberCols);
            Ellpack(NumberRows numberRows, NumberCols numberCols, Type type);
            Ellpack(Ellpack&& input) noexcept;

            ~Ellpack() = default;

            eilig::Ellpack Convert() const;

            bool IsUsed(Index row, Index col) const;
            bool IsUsed(Index row, Index col, Index& position) const;
            Index Add(Index row, Index col);
            void Remove(Index row, Index col);

            void Resize(NumberRows numberRows, NumberCols numberCols);
            void Resize(NumberRows numberRows, NumberCols numberCols, Scalar value);
            void Fill(Scalar value);
            void Init(eilig::Ellpack& input);
            void Dump() const;

            EntryProxy operator()(Index row, Index col);

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
            Scalar Trace() const;
            Scalar Sum() const;
            Ellpack Transpose() const;
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
            void Replace(Index row1, Index col1, const Ellpack& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            NumberCols GetWidth() const;
            Scalar GetValue(Index row, Index col) const;
            KernelPtr GetKernel() const;
            BufferPtr GetCountGPU() const;
            BufferPtr GetPositionGPU() const;
            BufferPtr GetDataGPU() const;

            void Equal(Index row, Index col, Scalar value);
            void Equal(Scalar value);
            void Equal(const Ellpack& value);
            void Equal(const std::initializer_list<std::initializer_list<Scalar>>& value);

            void Add(Scalar value);
            void Add(const Ellpack& value);
            void Sub(Scalar value);
            void Sub(const Ellpack& value);
            void Mul(Scalar value);
            void Mul(const Ellpack& rhs);

        private:
            void SetKernel(KernelPtr kernel);

            void Expand(NumberCols width);
            void Shrink();
            Index GrowthRate();
            Index MaxCount();
            void ShiftRight(Index row, Index position);
            void ShiftLeft(Index row, Index position);
            void Clear();
            Index FindWidthTranspose() const;
            
            NumberRows numberRows_{ 0 };
            NumberCols numberCols_{ 0 };
            NumberCols width_{ 0 };

            KernelPtr kernel_{ nullptr };
            BufferPtr countGPU_{ nullptr };
            BufferPtr positionGPU_{ nullptr };
            BufferPtr dataGPU_{ nullptr };
        };
    }

} /* namespace eilig */

#endif /* EILIG_OPENCL_MATRIX_ELLPACK_HPP_ */