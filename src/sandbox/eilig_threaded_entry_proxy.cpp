#include "eilig_threaded_entry_proxy.hpp"

namespace eilig
{
    namespace threaded
    {
		EntryProxyVector::EntryProxyVector(const Blocks& blocks, Index row) : blocks_(blocks), row_(row)
        {
        }
        Scalar EntryProxyVector::operator()()
        {
            return Read();
        }
        EntryProxyVector& EntryProxyVector::operator=(Scalar rhs)
        {
            Write(rhs);

            return *this;
        }
        EntryProxyVector& EntryProxyVector::operator+=(Scalar rhs)
        {
            Scalar aux = Read();

            Write(aux + rhs);

            return *this;
        }
        EntryProxyVector& EntryProxyVector::operator-=(Scalar rhs)
        {
            Scalar aux = Read();

            Write(aux - rhs);

            return *this;
        }
        void EntryProxyVector::SetBlockIndex(Index blockIndex)
        {
            if (blockIndex >= blocks_.size())
            {
				throw std::out_of_range("Block index is out of range.");
            }

			blockIndex_ = blockIndex;
        }
        Scalar EntryProxyVector::Read() const
        {
            Scalar res{ 0.0 };

            if (blocks_[blockIndex_].device.accelerator == Accelerator::cpu)
            {
				res = std::static_pointer_cast<VectorKernelCPU>(blocks_[blockIndex_].kernel)->GetVector().GetValue(row_);
            }
            else if (blocks_[blockIndex_].device.accelerator == Accelerator::gpu)
            {
                res = std::static_pointer_cast<VectorKernelGPU>(blocks_[blockIndex_].kernel)->GetVector().GetValue(row_);
            }

            return res;
        }
        void EntryProxyVector::Write(Scalar value)
        {
            for(Index i = 0; i < blocks_.size(); ++i)
            {
                if (blocks_[i].device.accelerator == Accelerator::cpu)
                {
                    std::static_pointer_cast<VectorKernelCPU>(blocks_[i].kernel)->GetVector().Equal(row_, value);
                }
                else if (blocks_[i].device.accelerator == Accelerator::gpu)
                {
                    std::static_pointer_cast<VectorKernelGPU>(blocks_[i].kernel)->GetVector().Equal(row_, value);
                }
			}
        }
    
        EntryProxyEllpack::EntryProxyEllpack(const Block& block, Index row, Index col) : block_(block), row_(row), col_(col)
        {
        }
        Scalar EntryProxyEllpack::operator()()
        {
            return Read();
        }
        EntryProxyEllpack& EntryProxyEllpack::operator=(Scalar rhs)
        {
            Write(rhs);

            return *this;
        }
        EntryProxyEllpack& EntryProxyEllpack::operator+=(Scalar rhs)
        {
            Scalar aux = Read();

            Write(aux + rhs);

            return *this;
        }
        EntryProxyEllpack& EntryProxyEllpack::operator-=(Scalar rhs)
        {
            Scalar aux = Read();

            Write(aux - rhs);

            return *this;
        }
        Scalar EntryProxyEllpack::Read() const
        {
            Scalar res{ 0.0 };		

            if (block_.device.accelerator == Accelerator::cpu)
            {
                res = std::static_pointer_cast<EllpackKernelCPU>(block_.kernel)->GetMatrix().GetValue(row_, col_);
            }
            else if (block_.device.accelerator == Accelerator::gpu)
            {
                res = std::static_pointer_cast<EllpackKernelGPU>(block_.kernel)->GetMatrix().GetValue(row_, col_);
            }

            return res;
        }
        void EntryProxyEllpack::Write(Scalar value)
        {
            if (block_.device.accelerator == Accelerator::cpu)
            {
                std::static_pointer_cast<EllpackKernelCPU>(block_.kernel)->GetMatrix().Equal(row_, col_, value);
            }
            else if (block_.device.accelerator == Accelerator::gpu)
            {
                std::static_pointer_cast<EllpackKernelGPU>(block_.kernel)->GetMatrix().Equal(row_, col_, value);
            }
        }
    }
} /* namespace eilig */