#ifndef EILIG_OPENCL_KERNELS_HPP_
#define EILIG_OPENCL_KERNELS_HPP_

#include "eilig_types.hpp"

namespace eilig
{
    namespace opencl
    {
        using BufferPtr = club::BufferPtr;

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
        static const String kEllpackMulS = "EllpackMulS";
        static const String kEllpackMulV = "EllpackMulV";
        static const String kEllpackMulM = "EllpackMulM";
        static const String kEllpackSwapRows = "EllpackSwapRows";
        static const String kEllpackSwapCols = "EllpackSwapCols";
        static const String kEllpackTranspose = "EllpackTranspose";
        static const String kEllpackFindWidthTranspose = "EllpackFindWidthTranspose";
        static const String kEllpackDiagonal = "EllpackDiagonal";
        static const String kEllpackRegion = "EllpackRegion";
        static const String kEllpackLower1 = "EllpackLower1";
        static const String kEllpackLower2 = "EllpackLower2";
        static const String kEllpackUpper1 = "EllpackUpper1";
        static const String kEllpackUpper2 = "EllpackUpper2";

        KernelsPtr CreateKernels(const String& fileName, const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber);

        class Kernels
        {
        public:
            Kernels() = default;
            virtual ~Kernels() = default;
            void Init(const String& fileName, const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber);

            club::PlatformPtr platform_{ nullptr };
            club::ContextPtr context_{ nullptr };
            club::ProgramPtr program_{ nullptr };

            club::KernelPtr kVectorCopyS_{ nullptr };
            club::KernelPtr kVectorAddS_{ nullptr };
            club::KernelPtr kVectorAddSl_{ nullptr };
            club::KernelPtr kVectorAddV_{ nullptr };
            club::KernelPtr kVectorPlus_{ nullptr };
            club::KernelPtr kVectorSubS_{ nullptr };
            club::KernelPtr kVectorSubSl_{ nullptr };
            club::KernelPtr kVectorSubV_{ nullptr };
            club::KernelPtr kVectorMinus_{ nullptr };
            club::KernelPtr kVectorMulS_{ nullptr };
            club::KernelPtr kVectorDot_{ nullptr };
            club::KernelPtr kVectorNormMax_{ nullptr };
            club::KernelPtr kVectorNormP_{ nullptr };
            club::KernelPtr kVectorNormP2_{ nullptr };
            club::KernelPtr kEllpackNormP_{ nullptr };
            club::KernelPtr kEllpackNormP2_{ nullptr };
            club::KernelPtr kEllpackMaxCount_{ nullptr };
            club::KernelPtr kEllpackExpandPosition_{ nullptr };
            club::KernelPtr kEllpackExpandData_{ nullptr };
            club::KernelPtr kEllpackShrinkPosition_{ nullptr };
            club::KernelPtr kEllpackShrinkData_{ nullptr };
            club::KernelPtr kEllpackCopyS_{ nullptr };
            club::KernelPtr kEllpackAddS_{ nullptr };
            club::KernelPtr kEllpackAddSl_{ nullptr };
            club::KernelPtr kEllpackPlus_{ nullptr };
            club::KernelPtr kEllpackSubS_{ nullptr };
            club::KernelPtr kEllpackSubSl_{ nullptr };
            club::KernelPtr kEllpackMinus_{ nullptr };
            club::KernelPtr kEllpackMulS_{ nullptr };
            club::KernelPtr kEllpackMulV_{ nullptr };
            club::KernelPtr kEllpackMulM_{ nullptr };
            club::KernelPtr kEllpackSwapRows_{ nullptr };
            club::KernelPtr kEllpackSwapCols_{ nullptr };
            club::KernelPtr kEllpackTranspose_{ nullptr };
            club::KernelPtr kEllpackFindWidthTranspose_{ nullptr };
            club::KernelPtr kEllpackDiagonal_{ nullptr };
            club::KernelPtr kEllpackRegion_{ nullptr };
            club::KernelPtr kEllpackLower1_{ nullptr };
            club::KernelPtr kEllpackLower2_{ nullptr };
            club::KernelPtr kEllpackUpper1_{ nullptr };
            club::KernelPtr kEllpackUpper2_{ nullptr };

        protected:
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