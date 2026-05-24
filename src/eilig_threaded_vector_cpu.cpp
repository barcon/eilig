#include "eilig_threaded_vector_cpu.hpp"
#include "eilig_vector.hpp"

namespace eilig
{
    namespace threaded
    {
        VectorCPUPtr CreateVectorCPU()
        {
            auto res = VectorCPU::Create();

            return res;
        }

        VectorCPUPtr VectorCPU::Create()
        {
            class MakeSharedEnabler : public VectorCPU
            {
            };

            auto res = std::make_shared<MakeSharedEnabler>();

            return res;
        }    
        Type VectorCPU::GetType() const
        {
            return type_;
        }
        eilig::Vector& VectorCPU::GetVector()
        {
            return vector_;
        }
        KernelVectorResize VectorCPU::GetKernelResize(NumberRows numberRows)
        {
            return KernelVectorResize(vector_, numberRows);
        }
        KernelVectorCopyScalar VectorCPU::GetKernelCopyScalar(Scalar value)
        {
            return KernelVectorCopyScalar(vector_, value);
        }
        KernelVectorCopyVector VectorCPU::GetKernelCopyVector(const eilig::Vector& vector)
        {
            return KernelVectorCopyVector(vector);
        }
        KernelVectorAddScalar VectorCPU::GetKernelAddScalar(Scalar value)
        {
            return KernelVectorAddScalar(vector_, value);
        }
        KernelVectorInitializerList VectorCPU::GetKernelVectorInitializerList(const std::initializer_list<Scalar>& values)
        {
            return KernelVectorInitializerList(vector_, values);
        }
    }
} /* namespace eilig */