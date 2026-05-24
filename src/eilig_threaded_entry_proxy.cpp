#include "eilig_threaded_entry_proxy.hpp"
#include "eilig_threaded_vector_cpu.hpp"

namespace eilig
{
    namespace threaded
    {
		EntryProxyVector::EntryProxyVector(Devices& devices, Index row) : devices_(devices), row_(row)
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
        void EntryProxyVector::SetDeviceIndex(Index deviceIndex)
        {
            if (deviceIndex >= devices_.size())
            {
				throw std::out_of_range("Device index is out of range.");
            }

			deviceIndex_ = deviceIndex;
        }
        Scalar EntryProxyVector::Read() const
        {
            Scalar res{ 0.0 };

            if (devices_[deviceIndex_]->GetType() == device_vector_cpu)
            {
				res = std::static_pointer_cast<VectorCPU>(devices_[deviceIndex_])->GetVector()(row_);
            }

            return res;
        }
        void EntryProxyVector::Write(Scalar value)
        {
            if (devices_[deviceIndex_]->GetType() == device_vector_cpu)
            {
                std::static_pointer_cast<VectorCPU>(devices_[deviceIndex_])->GetVector()(row_) = value;
            }
        }
    }
} /* namespace eilig */