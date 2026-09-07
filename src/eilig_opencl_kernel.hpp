#ifndef EILIG_OPENCL_KERNEL_HPP_
#define EILIG_OPENCL_KERNEL_HPP_

#include "eilig_types.hpp"

namespace eilig
{
    namespace opencl
    {
		void Initialize(const String& fileName, const club::PlatformIndex& platformIndex, const club::DeviceIndices& deviceIndices);
		
        ContextPtr GetContext();

        static const String kVectorCopyS = "VectorCopyS";
        static const String kVectorAddS = "VectorAddS";
        static const String kVectorAddV = "VectorAddV";
        static const String kVectorSubS = "VectorSubS";
        static const String kVectorSubV = "VectorSubV";
        static const String kVectorMulS = "VectorMulS";
        static const String kVectorDot = "VectorDot";
        static const String kVectorNormMax = "VectorNormMax";
        static const String kVectorNormP = "VectorNormP";
        static const String kVectorNormP2 = "VectorNormP2";
        
        KernelVectorPtr CreateKernelVector();

        class KernelVector
        {
        public:
            virtual ~KernelVector() = default;

            static KernelVectorPtr Create();

            club::KernelPtr kVectorCopyS_{ nullptr };
            club::KernelPtr kVectorAddS_{ nullptr };
            club::KernelPtr kVectorAddV_{ nullptr };
            club::KernelPtr kVectorSubS_{ nullptr };
            club::KernelPtr kVectorSubV_{ nullptr };
            club::KernelPtr kVectorMulS_{ nullptr };
            club::KernelPtr kVectorDot_{ nullptr };
            club::KernelPtr kVectorNormMax_{ nullptr };
            club::KernelPtr kVectorNormP_{ nullptr };
            club::KernelPtr kVectorNormP2_{ nullptr };

        protected:
            KernelVector();
            KernelVector(const KernelVector& copy) = delete;
            KernelVector(KernelVector&& move) = delete;

            KernelVector& operator=(const KernelVector& copy) = delete;
            KernelVector& operator=(KernelVector&& copy) = delete;
        };

        static const String kMatrixCopyS = "MatrixCopyS";
        static const String kMatrixAddS = "MatrixAddS";
        static const String kMatrixAddM = "MatrixAddM";
        static const String kMatrixSubS = "MatrixSubS";
        static const String kMatrixSubM = "MatrixSubM";
        static const String kMatrixMulScalar = "MatrixMulScalar";
        static const String kMatrixMulVector = "MatrixMulVector";
        static const String kMatrixMulMatrix = "MatrixMulMatrix";
        static const String kMatrixSwapRows = "MatrixSwapRows";
        static const String kMatrixSwapCols = "MatrixSwapCols";
        static const String kMatrixTranspose = "MatrixTranspose";
        static const String kMatrixDiagonal = "MatrixDiagonal";
        static const String kMatrixDiagonalScale = "MatrixDiagonalScale";
        static const String kMatrixDiagonalVector = "MatrixDiagonalVector";
        static const String kMatrixLower1 = "MatrixLower1";
        static const String kMatrixLower2 = "MatrixLower2";
        static const String kMatrixUpper1 = "MatrixUpper1";
        static const String kMatrixUpper2 = "MatrixUpper2";
        static const String kMatrixRegion = "MatrixRegion";
        static const String kMatrixTrace = "MatrixTrace";
        static const String kMatrixSum = "MatrixSum";

        KernelMatrixPtr CreateKernelMatrix();

        class KernelMatrix
        {
        public:
            virtual ~KernelMatrix() = default;

            static KernelMatrixPtr Create();

            club::KernelPtr kMatrixCopyS_{ nullptr };
            club::KernelPtr kMatrixAddS_{ nullptr };
            club::KernelPtr kMatrixAddM_{ nullptr };
            club::KernelPtr kMatrixSubS_{ nullptr };
            club::KernelPtr kMatrixSubM_{ nullptr };
            club::KernelPtr kMatrixMulScalar_{ nullptr };
            club::KernelPtr kMatrixMulVector_{ nullptr };
            club::KernelPtr kMatrixMulMatrix_{ nullptr };
            club::KernelPtr kMatrixSwapRows_{ nullptr };
            club::KernelPtr kMatrixSwapCols_{ nullptr };
            club::KernelPtr kMatrixTranspose_{ nullptr };
            club::KernelPtr kMatrixDiagonal_{ nullptr };
            club::KernelPtr kMatrixDiagonalScale_{ nullptr };
            club::KernelPtr kMatrixDiagonalVector_{ nullptr };
            club::KernelPtr kMatrixLower1_{ nullptr };
            club::KernelPtr kMatrixLower2_{ nullptr };
            club::KernelPtr kMatrixUpper1_{ nullptr };
            club::KernelPtr kMatrixUpper2_{ nullptr };
            club::KernelPtr kMatrixRegion_{ nullptr };
            club::KernelPtr kMatrixTrace_{ nullptr };
            club::KernelPtr kMatrixSum_{ nullptr };

        protected:
            KernelMatrix();
            KernelMatrix(const KernelMatrix& copy) = delete;
            KernelMatrix(KernelMatrix&& move) = delete;

            KernelMatrix& operator=(const KernelMatrix& copy) = delete;
            KernelMatrix& operator=(KernelMatrix&& copy) = delete;
        };

        static const String kEllpackNormP = "EllpackNormP";
        static const String kEllpackNormP2 = "EllpackNormP2";
        static const String kEllpackMaxCount = "EllpackMaxCount";
        static const String kEllpackExpandPosition = "EllpackExpandPosition";
        static const String kEllpackExpandData = "EllpackExpandData";
        static const String kEllpackShrinkPosition = "EllpackShrinkPosition";
        static const String kEllpackShrinkData = "EllpackShrinkData";
        static const String kEllpackCopyS = "EllpackCopyS";
        static const String kEllpackAddS = "EllpackAddS";
        static const String kEllpackSubS = "EllpackSubS";
        static const String kEllpackMulScalar = "EllpackMulScalar";
        static const String kEllpackMulVector = "EllpackMulVector";
        static const String kEllpackMulMatrix = "EllpackMulMatrix";
        static const String kEllpackSwapRows = "EllpackSwapRows";
        static const String kEllpackSwapCols = "EllpackSwapCols";
        static const String kEllpackTranspose = "EllpackTranspose";
        static const String kEllpackFindWidthTranspose = "EllpackFindWidthTranspose";
        static const String kEllpackDiagonal = "EllpackDiagonal";
        static const String kEllpackDiagonalScale = "EllpackDiagonalScale";
        static const String kEllpackDiagonalVector = "EllpackDiagonalVector";
        static const String kEllpackLower1 = "EllpackLower1";
        static const String kEllpackLower2 = "EllpackLower2";
        static const String kEllpackUpper1 = "EllpackUpper1";
        static const String kEllpackUpper2 = "EllpackUpper2";
        static const String kEllpackRegion = "EllpackRegion";
        static const String kEllpackTrace = "EllpackTrace";
        static const String kEllpackSum = "EllpackSum";

        KernelEllpackPtr CreateKernelEllpack();

        class KernelEllpack
        {
        public:
            virtual ~KernelEllpack() = default;

            static KernelEllpackPtr Create();

            club::KernelPtr kEllpackNormP_{ nullptr };
            club::KernelPtr kEllpackNormP2_{ nullptr };
            club::KernelPtr kEllpackMaxCount_{ nullptr };
            club::KernelPtr kEllpackExpandPosition_{ nullptr };
            club::KernelPtr kEllpackExpandData_{ nullptr };
            club::KernelPtr kEllpackShrinkPosition_{ nullptr };
            club::KernelPtr kEllpackShrinkData_{ nullptr };
            club::KernelPtr kEllpackCopyS_{ nullptr };
            club::KernelPtr kEllpackAddS_{ nullptr };
            club::KernelPtr kEllpackSubS_{ nullptr };
            club::KernelPtr kEllpackMulScalar_{ nullptr };
            club::KernelPtr kEllpackMulVector_{ nullptr };
            club::KernelPtr kEllpackMulMatrix_{ nullptr };
            club::KernelPtr kEllpackSwapRows_{ nullptr };
            club::KernelPtr kEllpackSwapCols_{ nullptr };
            club::KernelPtr kEllpackTranspose_{ nullptr };
            club::KernelPtr kEllpackFindWidthTranspose_{ nullptr };
            club::KernelPtr kEllpackDiagonal_{ nullptr };
            club::KernelPtr kEllpackDiagonalScale_{ nullptr };
            club::KernelPtr kEllpackDiagonalVector_{ nullptr };
            club::KernelPtr kEllpackLower1_{ nullptr };
            club::KernelPtr kEllpackLower2_{ nullptr };
            club::KernelPtr kEllpackUpper1_{ nullptr };
            club::KernelPtr kEllpackUpper2_{ nullptr };
            club::KernelPtr kEllpackRegion_{ nullptr };
            club::KernelPtr kEllpackTrace_{ nullptr };
            club::KernelPtr kEllpackSum_{ nullptr };

        protected:
            KernelEllpack();
            KernelEllpack(const KernelEllpack& copy) = delete;
            KernelEllpack(KernelEllpack&& move) = delete;

            KernelEllpack& operator=(const KernelEllpack& copy) = delete;
            KernelEllpack& operator=(KernelEllpack&& copy) = delete;
        };

    } // namespace opencl
} // namespace eilig

#endif