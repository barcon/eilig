#ifndef EILIG_OPENCL_MATRIX_ELLPACK_HPP_
#define EILIG_OPENCL_MATRIX_ELLPACK_HPP_

#include "eilig_types.hpp"
#include "eilig_matrix_ellpack.hpp"

#include "eilig_opencl_kernels.hpp"
#include "eilig_opencl_entry_proxy.hpp"
#include "eilig_opencl_vector.hpp"

namespace eilig
{
    namespace opencl
    {
        class Ellpack
        {
        public:
            Ellpack(KernelsPtr kernels);
            Ellpack(const Ellpack& input);
            Ellpack(KernelsPtr kernels, const std::vector<Scalars>& values);
            Ellpack(KernelsPtr kernels, const eilig::Matrix& input);
            Ellpack(KernelsPtr kernels, const eilig::Ellpack& input);
            Ellpack(KernelsPtr kernels, NumberRows numberRows, NumberCols numberCols);
            Ellpack(KernelsPtr kernels, NumberRows numberRows, NumberCols numberCols, Type type);
            Ellpack(Ellpack&& input) noexcept;

            ~Ellpack() = default;

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
            Ellpack Transpose() const;
            Ellpack Diagonal() const;
            Ellpack Lower(bool diag) const;
            Ellpack LowerWithDiagonal() const;
            Ellpack LowerWithoutDiagonal() const;
            Ellpack Upper(bool diag) const;
            Ellpack UpperWithDiagonal() const;
            Ellpack UpperWithoutDiagonal() const;
            Ellpack Region(Index row1, Index col1, Index row2, Index col2) const;
            void Region(Index row1, Index col1, Index row2, Index col2, const Ellpack& in);

            NumberRows GetRows() const;
            NumberCols GetCols() const;
            NumberCols GetWidth() const;
            Scalar GetValue(Index row, Index col) const;
            KernelsPtr GetKernels() const;
            BufferPtr GetCountGPU() const;
            BufferPtr GetPositionGPU() const;
            BufferPtr GetDataGPU() const;

            void SetValue(Index row, Index col, Scalar value);

        private:

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

            KernelsPtr kernels_{ nullptr };
            BufferPtr countGPU_{ nullptr };
            BufferPtr positionGPU_{ nullptr };
            BufferPtr dataGPU_{ nullptr };
        };
    }

} /* namespace eilig */

#endif /* EILIG_OPENCL_MATRIX_ELLPACK_HPP_ */