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

            explicit EntryProxy(club::BufferPtr buffer, Index offset);

            void SetBuffer(club::BufferPtr buffer);
            void SetIndex(Index index);

            Scalar operator()();
            EntryProxy& operator=(Scalar rhs);
            EntryProxy& operator+=(Scalar rhs);
            EntryProxy& operator-=(Scalar rhs);
           
            ~EntryProxy() = default;

        private:

            Index index_{ 0 };
            club::BufferPtr buffer_{ nullptr };

            Scalar Read() const;
            void Write(Scalar value);
        };
    }

} /* namespace eilig */

#endif /* EILIG_OPENCL_ENTRY_PROXY_HPP_ */