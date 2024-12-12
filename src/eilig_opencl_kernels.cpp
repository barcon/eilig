#ifdef EILIG_ENABLE_OPENCL

#include "eilig_opencl_kernels.hpp"

namespace eilig
{
    namespace opencl
    {
        KernelsPtr CreateKernels(const String& fileName, const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber)
        {
            auto res = new Kernels();

            res->Init(fileName, platformNumber, deviceNumber);

            return res;
        }
        void Kernels::Init(const String& fileName, const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber)
        {
            InitPlatform();
            InitContext(platformNumber, deviceNumber);
            InitProgram(fileName);
            InitKernels();
        }
        void Kernels::InitPlatform()
        {
            platform_ = club::CreatePlatform();

            if (!platform_)
            {
                logger::Error(headerEilig, "Platforms not initialized");
                return;
            }
        }
        void Kernels::InitContext(const club::PlatformNumber& platformNumber, const club::DeviceNumber& deviceNumber)
        {
            context_ = club::CreateContext(platform_, platformNumber, deviceNumber);
            if (!context_)
            {
                logger::Error(headerEilig, "Context (%d)(%d) could not be created", platformNumber, deviceNumber);
                return;
            }

            auto contextInfo = context_->GetInfo();
        }
        void Kernels::InitProgram(const String& fileName)
        {
            program_ = club::CreateProgramFromFile(context_, fileName);
            if (!program_)
            {
                logger::Error(headerEilig, "Program could not be created " + fileName);
                return;
            }
        }
        void Kernels::InitKernels()
        {
            kVectorCopyS_ = club::CreateKernel(program_, kVectorCopyS, 1);
            kVectorAddS_ = club::CreateKernel(program_, kVectorAddS, 1);
            kVectorAddSl_ = club::CreateKernel(program_, kVectorAddSl, 1);
            kVectorAddV_ = club::CreateKernel(program_, kVectorAddV, 1);
            kVectorPlus_ = club::CreateKernel(program_, kVectorPlus, 1);
            kVectorSubS_ = club::CreateKernel(program_, kVectorSubS, 1);
            kVectorSubSl_ = club::CreateKernel(program_, kVectorSubSl, 1);
            kVectorSubV_ = club::CreateKernel(program_, kVectorSubV, 1);
            kVectorMinus_ = club::CreateKernel(program_, kVectorMinus, 1);
            kVectorMulS_ = club::CreateKernel(program_, kVectorMulS, 1);
            kVectorDot_ = club::CreateKernel(program_, kVectorDot, 1);
            kVectorNormMax_ = club::CreateKernel(program_, kVectorNormMax, 1);
            kVectorNormP_ = club::CreateKernel(program_, kVectorNormP, 1);
            kVectorNormP2_ = club::CreateKernel(program_, kVectorNormP2, 1);
            kEllpackNormP_ = club::CreateKernel(program_, kEllpackNormP, 1);
            kEllpackNormP2_ = club::CreateKernel(program_, kEllpackNormP2, 1);
            kEllpackMaxCount_ = club::CreateKernel(program_, kEllpackMaxCount, 1);
            kEllpackExpandPosition_ = club::CreateKernel(program_, kEllpackExpandPosition, 2);
            kEllpackExpandData_ = club::CreateKernel(program_, kEllpackExpandData, 2);
            kEllpackShrinkPosition_ = club::CreateKernel(program_, kEllpackShrinkPosition, 2);
            kEllpackShrinkData_ = club::CreateKernel(program_, kEllpackShrinkData, 2);
            kEllpackCopyS_ = club::CreateKernel(program_, kEllpackCopyS, 2);
            kEllpackAddS_ = club::CreateKernel(program_, kEllpackAddS, 2);
            kEllpackAddSl_ = club::CreateKernel(program_, kEllpackAddSl, 2);
            kEllpackPlus_ = club::CreateKernel(program_, kEllpackPlus, 2);
            kEllpackSubS_ = club::CreateKernel(program_, kEllpackSubS, 2);
            kEllpackSubSl_ = club::CreateKernel(program_, kEllpackSubSl, 2);
            kEllpackMinus_ = club::CreateKernel(program_, kEllpackMinus, 2);
            kEllpackMulS_ = club::CreateKernel(program_, kEllpackMulS, 2);
            kEllpackMulV_ = club::CreateKernel(program_, kEllpackMulV, 1);
            kEllpackMulM_ = club::CreateKernel(program_, kEllpackMulM, 1);
            kEllpackSwapRows_ = club::CreateKernel(program_, kEllpackSwapRows, 1);
            kEllpackSwapCols_ = club::CreateKernel(program_, kEllpackSwapCols, 1);
            kEllpackTranspose_ = club::CreateKernel(program_, kEllpackTranspose, 1);
            kEllpackFindWidthTranspose_ = club::CreateKernel(program_, kEllpackFindWidthTranspose, 2);
            kEllpackDiagonal_ = club::CreateKernel(program_, kEllpackDiagonal, 1);
            kEllpackRegion_ = club::CreateKernel(program_, kEllpackRegion, 1);
            kEllpackLower1_ = club::CreateKernel(program_, kEllpackLower1, 1);
            kEllpackLower2_ = club::CreateKernel(program_, kEllpackLower2, 1);
            kEllpackUpper1_ = club::CreateKernel(program_, kEllpackUpper1, 1);
            kEllpackUpper2_ = club::CreateKernel(program_, kEllpackUpper2, 1);
        }
    }
} /* namespace eilig */

#endif