#ifndef EILIG_THREADED_ENTRY_PROXY_HPP_
#define EILIG_THREADED_ENTRY_PROXY_HPP_

#include "eilig_threaded.hpp"

namespace eilig
{
    namespace threaded
    {
        class EntryProxyVector
        {
        public:
            explicit EntryProxyVector(const Blocks& blocks, Index row);

            ~EntryProxyVector() = default;
            
            Scalar operator()();
            EntryProxyVector& operator=(Scalar rhs);
            EntryProxyVector& operator+=(Scalar rhs);
            EntryProxyVector& operator-=(Scalar rhs);
            
            void SetBlockIndex(Index kernelIndex);
                     
        private:
            void Write(Scalar value);
            Scalar Read() const;

            Index row_{ 0 };
			Index blockIndex_{ 0 };

            const Blocks& blocks_;
        };

        class EntryProxyEllpack
        {
        public:
            explicit EntryProxyEllpack(const Block& block, Index row, Index col);

            ~EntryProxyEllpack() = default;

            Scalar operator()();
            EntryProxyEllpack& operator=(Scalar rhs);
            EntryProxyEllpack& operator+=(Scalar rhs);
            EntryProxyEllpack& operator-=(Scalar rhs);

            void SetKernelIndex(Index kernelIndex);

        private:
            void Write(Scalar value);
            Scalar Read() const;

            Index row_{ 0 };
            Index col_{ 0 };
            Index kernelIndex_{ 0 };

            const Block& block_;
        };
    }

} /* namespace eilig */

#endif /* EILIG_THREADED_ENTRY_PROXY_HPP_ */