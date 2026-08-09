#ifndef EILIG_OPENCL_KERNELS_HPP_
#define EILIG_OPENCL_KERNELS_HPP_

#include "eilig_types.hpp"

namespace eilig
{
    namespace opencl
    {
        using BufferPtr = club::BufferPtr;
		using KernelPtr = club::KernelPtr;

        static const String kVectorCopyS = "VectorCopyS";
        static const String kVectorAddS = "VectorAddS";
        static const String kVectorAddSl = "VectorAddSl";
        static const String kVectorAddV = "VectorAddV";
        static const String kVectorPlus = "VectorPlus";
        static const String kVectorSubS = "VectorSubS";
        static const String kVectorSubSl = "VectorSubSl";
        static const String kVectorSubV = "VectorSubV";
        static const String kVectorMinus = "VectorMinus";
        static const String kVectorMulS = "VectorMulS";
        static const String kVectorDot = "VectorDot";
        static const String kVectorNormMax = "VectorNormMax";
        static const String kVectorNormP = "VectorNormP";
        static const String kVectorNormP2 = "VectorNormP2";
        static const String kEllpackNormP = "EllpackNormP";
        static const String kEllpackNormP2 = "EllpackNormP2";
        static const String kEllpackMaxCount = "EllpackMaxCount";
        static const String kEllpackExpandPosition = "EllpackExpandPosition";
        static const String kEllpackExpandData = "EllpackExpandData";
        static const String kEllpackShrinkPosition = "EllpackShrinkPosition";
        static const String kEllpackShrinkData = "EllpackShrinkData";
        static const String kEllpackCopyS = "EllpackCopyS";
        static const String kEllpackAddS = "EllpackAddS";
        static const String kEllpackAddSl = "EllpackAddSl";
        static const String kEllpackPlus = "EllpackPlus";
        static const String kEllpackSubS = "EllpackSubS";
        static const String kEllpackSubSl = "EllpackSubSl";
        static const String kEllpackMinus = "EllpackMinus";
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

        KernelsPtr CreateKernels(const String& fileName, const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber);

        class Kernels
        {
        public:
            virtual ~Kernels() = default;

            static KernelsPtr Create();

            void Init(const String& fileName, const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber);

            club::PlatformPtr platform_{ nullptr };
            club::ContextPtr context_{ nullptr };
            club::ProgramPtr program_{ nullptr };

            KernelPtr kVectorCopyS_{ nullptr };
            KernelPtr kVectorAddS_{ nullptr };
            KernelPtr kVectorAddSl_{ nullptr };
            KernelPtr kVectorAddV_{ nullptr };
            KernelPtr kVectorPlus_{ nullptr };
            KernelPtr kVectorSubS_{ nullptr };
            KernelPtr kVectorSubSl_{ nullptr };
            KernelPtr kVectorSubV_{ nullptr };
            KernelPtr kVectorMinus_{ nullptr };
            KernelPtr kVectorMulS_{ nullptr };
            KernelPtr kVectorDot_{ nullptr };
            KernelPtr kVectorNormMax_{ nullptr };
            KernelPtr kVectorNormP_{ nullptr };
            KernelPtr kVectorNormP2_{ nullptr };
            KernelPtr kEllpackNormP_{ nullptr };
            KernelPtr kEllpackNormP2_{ nullptr };
            KernelPtr kEllpackMaxCount_{ nullptr };
            KernelPtr kEllpackExpandPosition_{ nullptr };
            KernelPtr kEllpackExpandData_{ nullptr };
            KernelPtr kEllpackShrinkPosition_{ nullptr };
            KernelPtr kEllpackShrinkData_{ nullptr };
            KernelPtr kEllpackCopyS_{ nullptr };
            KernelPtr kEllpackAddS_{ nullptr };
            KernelPtr kEllpackAddSl_{ nullptr };
            KernelPtr kEllpackPlus_{ nullptr };
            KernelPtr kEllpackSubS_{ nullptr };
            KernelPtr kEllpackSubSl_{ nullptr };
            KernelPtr kEllpackMinus_{ nullptr };
            KernelPtr kEllpackMulScalar_{ nullptr };
            KernelPtr kEllpackMulVector_{ nullptr };
            KernelPtr kEllpackMulMatrix_{ nullptr };
            KernelPtr kEllpackSwapRows_{ nullptr };
            KernelPtr kEllpackSwapCols_{ nullptr };
            KernelPtr kEllpackTranspose_{ nullptr };
            KernelPtr kEllpackFindWidthTranspose_{ nullptr };
            KernelPtr kEllpackDiagonal_{ nullptr };
            KernelPtr kEllpackDiagonalScale_{ nullptr };
            KernelPtr kEllpackDiagonalVector_{ nullptr };
            KernelPtr kEllpackLower1_{ nullptr };
            KernelPtr kEllpackLower2_{ nullptr };
            KernelPtr kEllpackUpper1_{ nullptr };
            KernelPtr kEllpackUpper2_{ nullptr };
            KernelPtr kEllpackRegion_{ nullptr };
            KernelPtr kEllpackTrace_{ nullptr };
            KernelPtr kEllpackSum_{ nullptr };

        protected:
            Kernels() = default;
            Kernels(const Kernels& copy) = delete;
            Kernels(Kernels&& move) = delete;

            Kernels& operator=(const Kernels& copy) = delete;
            Kernels& operator=(Kernels&& copy) = delete;

            void InitPlatform();
            void InitContext(const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber);
            void InitProgram(const String& fileName);
            void InitKernels();
        };

    } // namespace opencl
} // namespace eilig

#endif