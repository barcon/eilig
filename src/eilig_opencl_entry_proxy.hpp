#ifndef EILIG_OPENCL_ENTRY_PROXY_HPP_
#define EILIG_OPENCL_ENTRY_PROXY_HPP_

#include "eilig_types.hpp"

namespace eilig
{
    namespace opencl
    {
        class EntryProxy
        {
        public:
            explicit EntryProxy(club::BufferPtr buffer, Index offset, const DeviceIndex& deviceIndex);

            ~EntryProxy() = default;

            Scalar operator()();
            EntryProxy& operator=(Scalar rhs);
            EntryProxy& operator+=(Scalar rhs);
            EntryProxy& operator-=(Scalar rhs);
           
        private:
            void SetBuffer(club::BufferPtr buffer);
            void SetIndex(Index index);
            void Write(Scalar value);
            Scalar Read() const;

            Index index_{ 0 };
            club::BufferPtr buffer_{ nullptr };

			const DeviceIndex& deviceIndex_;
        };
    }

} /* namespace eilig */

#endif /* EILIG_OPENCL_ENTRY_PROXY_HPP_ */