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
            explicit EntryProxyVector(Devices& devices, Index row);

            ~EntryProxyVector() = default;
            
            Scalar operator()();
            EntryProxyVector& operator=(Scalar rhs);
            EntryProxyVector& operator+=(Scalar rhs);
            EntryProxyVector& operator-=(Scalar rhs);
                     
        private:
            void SetDeviceIndex(Index deviceIndex);
            void Write(Scalar value);
            Scalar Read() const;

            Index row_{ 0 };
			Index deviceIndex_{ 0 };

            Devices& devices_;
        };
    }

} /* namespace eilig */

#endif /* EILIG_THREADED_ENTRY_PROXY_HPP_ */