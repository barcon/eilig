#include "eilig_opencl_kernel.hpp"

namespace eilig
{
    namespace opencl
    {
        static PlatformPtr platform{ nullptr };
        static ContextPtr context{ nullptr };
        static ProgramPtr program{ nullptr };

        void InitPlatform()
        {
            platform = club::CreatePlatform();

            if (!platform)
            {
				throw std::runtime_error("Platform could not be created");
            }
        }
        void InitContext(const club::PlatformIndex& platformIndex, const club::DeviceIndices& deviceIndices)
        {
            context = club::CreateContext(platform, platformIndex, deviceIndices);

            if (!context)
            {
                throw std::runtime_error(utils::string::Format("Context ({}) could not be created", platformIndex));
            }
        }
        void InitProgram(const String& fileName)
        {
            program = club::CreateProgramFromFile(context, fileName);

            if (!program)
            {
                throw std::runtime_error(utils::string::Format("Program could not be created {}", fileName));
            }
        }
        void Initialize(const String& fileName, const club::PlatformIndex& platformIndex, const club::DeviceIndices& deviceIndices)
        {
            InitPlatform();
            InitContext(platformIndex, deviceIndices);
            InitProgram(fileName);
        }

        ContextPtr GetContext()
        {
            return context;
        }

        KernelVectorPtr CreateKernelVector()
        {
            auto res = KernelVector::Create();

            return res;
        }
        KernelVectorPtr KernelVector::Create()
        {
            class MakeSharedEnabler : public KernelVector
            {
            };

            auto res = std::make_shared<MakeSharedEnabler>();

            return res;
        }
        KernelVector::KernelVector()
        {
            kVectorCopyS_ = club::CreateKernel(program, kVectorCopyS, 1);
            kVectorAddS_ = club::CreateKernel(program, kVectorAddS, 1);
            kVectorAddV_ = club::CreateKernel(program, kVectorAddV, 1);
            kVectorSubS_ = club::CreateKernel(program, kVectorSubS, 1);
            kVectorSubV_ = club::CreateKernel(program, kVectorSubV, 1);
            kVectorMulS_ = club::CreateKernel(program, kVectorMulS, 1);
            kVectorDot_ = club::CreateKernel(program, kVectorDot, 1);
            kVectorNormMax_ = club::CreateKernel(program, kVectorNormMax, 1);
            kVectorNormP_ = club::CreateKernel(program, kVectorNormP, 1);
            kVectorNormP2_ = club::CreateKernel(program, kVectorNormP2, 1);
        }
    
        KernelMatrixPtr CreateKernelMatrix()
        {
            auto res = KernelMatrix::Create();

            return res;
        }
        KernelMatrixPtr KernelMatrix::Create()
        {
            class MakeSharedEnabler : public KernelMatrix
            {
            };

            auto res = std::make_shared<MakeSharedEnabler>();

            return res;
        }
        KernelMatrix::KernelMatrix()
        {
            kMatrixCopyS_ = club::CreateKernel(program, kMatrixCopyS, 2);
            kMatrixAddS_ = club::CreateKernel(program, kMatrixAddS, 2);
            kMatrixAddM_ = club::CreateKernel(program, kMatrixAddM, 2);
            kMatrixSubS_ = club::CreateKernel(program, kMatrixSubS, 2);
            kMatrixSubM_ = club::CreateKernel(program, kMatrixSubM, 2);
            kMatrixMulScalar_ = club::CreateKernel(program, kMatrixMulScalar, 2);
            kMatrixMulVector_ = club::CreateKernel(program, kMatrixMulVector, 1);
            kMatrixMulMatrix_ = club::CreateKernel(program, kMatrixMulMatrix, 1);
            kMatrixSwapRows_ = club::CreateKernel(program, kMatrixSwapRows, 1);
            kMatrixSwapCols_ = club::CreateKernel(program, kMatrixSwapCols, 1);
            kMatrixTranspose_ = club::CreateKernel(program, kMatrixTranspose, 1);
            kMatrixDiagonal_ = club::CreateKernel(program, kMatrixDiagonal, 1);
            kMatrixDiagonalScale_ = club::CreateKernel(program, kMatrixDiagonalScale, 1);
            kMatrixDiagonalVector_ = club::CreateKernel(program, kMatrixDiagonalVector, 1);
            kMatrixLower1_ = club::CreateKernel(program, kMatrixLower1, 1);
            kMatrixLower2_ = club::CreateKernel(program, kMatrixLower2, 1);
            kMatrixUpper1_ = club::CreateKernel(program, kMatrixUpper1, 1);
            kMatrixUpper2_ = club::CreateKernel(program, kMatrixUpper2, 1);
            kMatrixRegion_ = club::CreateKernel(program, kMatrixRegion, 1);
            kMatrixTrace_ = club::CreateKernel(program, kMatrixTrace, 1);
            kMatrixSum_ = club::CreateKernel(program, kMatrixSum, 1);
        }

        KernelEllpackPtr CreateKernelEllpack()
        {
            auto res = KernelEllpack::Create();

            return res;
        }
        KernelEllpackPtr KernelEllpack::Create()
        {
            class MakeSharedEnabler : public KernelEllpack
            {
            };

            auto res = std::make_shared<MakeSharedEnabler>();

            return res;
        }
        KernelEllpack::KernelEllpack()
        {
            kEllpackNormP_ = club::CreateKernel(program, kEllpackNormP, 1);
            kEllpackNormP2_ = club::CreateKernel(program, kEllpackNormP2, 1);
            kEllpackMaxCount_ = club::CreateKernel(program, kEllpackMaxCount, 1);
            kEllpackExpandPosition_ = club::CreateKernel(program, kEllpackExpandPosition, 2);
            kEllpackExpandData_ = club::CreateKernel(program, kEllpackExpandData, 2);
            kEllpackShrinkPosition_ = club::CreateKernel(program, kEllpackShrinkPosition, 2);
            kEllpackShrinkData_ = club::CreateKernel(program, kEllpackShrinkData, 2);
            kEllpackCopyS_ = club::CreateKernel(program, kEllpackCopyS, 2);
            kEllpackAddS_ = club::CreateKernel(program, kEllpackAddS, 2);
            kEllpackSubS_ = club::CreateKernel(program, kEllpackSubS, 2);
            kEllpackMulScalar_ = club::CreateKernel(program, kEllpackMulScalar, 2);
            kEllpackMulVector_ = club::CreateKernel(program, kEllpackMulVector, 1);
            kEllpackMulMatrix_ = club::CreateKernel(program, kEllpackMulMatrix, 1);
            kEllpackSwapRows_ = club::CreateKernel(program, kEllpackSwapRows, 1);
            kEllpackSwapCols_ = club::CreateKernel(program, kEllpackSwapCols, 1);
            kEllpackTranspose_ = club::CreateKernel(program, kEllpackTranspose, 1);
            kEllpackFindWidthTranspose_ = club::CreateKernel(program, kEllpackFindWidthTranspose, 2);
            kEllpackDiagonal_ = club::CreateKernel(program, kEllpackDiagonal, 1);
            kEllpackDiagonalScale_ = club::CreateKernel(program, kEllpackDiagonalScale, 1);
            kEllpackDiagonalVector_ = club::CreateKernel(program, kEllpackDiagonalVector, 1);
            kEllpackLower1_ = club::CreateKernel(program, kEllpackLower1, 1);
            kEllpackLower2_ = club::CreateKernel(program, kEllpackLower2, 1);
            kEllpackUpper1_ = club::CreateKernel(program, kEllpackUpper1, 1);
            kEllpackUpper2_ = club::CreateKernel(program, kEllpackUpper2, 1);
            kEllpackRegion_ = club::CreateKernel(program, kEllpackRegion, 1);
            kEllpackTrace_ = club::CreateKernel(program, kEllpackTrace, 1);
            kEllpackSum_ = club::CreateKernel(program, kEllpackSum, 1);
        }
    }
} /* namespace eilig */