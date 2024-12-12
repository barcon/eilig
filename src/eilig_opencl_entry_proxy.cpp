#ifdef EILIG_ENABLE_OPENCL

#include "eilig_opencl_entry_proxy.hpp"

namespace eilig
{
    namespace opencl
    {
        EntryProxy::EntryProxy(club::BufferPtr buffer, Index index)
        {
            SetBuffer(buffer);
            SetIndex(index);
        }
        void EntryProxy::SetBuffer(club::BufferPtr buffer)
        {
            buffer_ = buffer;
        }
        void EntryProxy::SetIndex(Index index)
        {
            index_ = index;
        }
        Scalar EntryProxy::operator()()
        {
            return Read();
        }
        EntryProxy& EntryProxy::operator=(Scalar rhs)
        {
            Write(rhs);

            return *this;
        }
        EntryProxy& EntryProxy::operator+=(Scalar rhs)
        {
            Scalar aux = Read();

            Write(aux + rhs);

            return *this;
        }
        EntryProxy& EntryProxy::operator-=(Scalar rhs)
        {
            Scalar aux = Read();

            Write(aux - rhs);

            return *this;
        }
        Scalar EntryProxy::Read() const
        {
            Scalar res{ 0.0 };

            buffer_->Read(sizeof(Scalar) * index_, sizeof(Scalar), &res, CL_TRUE);

            return res;
        }
        void EntryProxy::Write(Scalar value)
        {
            buffer_->Write(sizeof(Scalar) * index_, sizeof(Scalar), &value, CL_TRUE);
        }
    }
} /* namespace eilig */

#endif