#include "eilig_opencl_vector.hpp"

namespace eilig
{
    namespace opencl
    {
        Vector::Vector()
        {
            InitKernel();
            Resize(1);
        }
        Vector::Vector(const Vector& input)
        {
            (*this) = input;
        }
        Vector::Vector(const std::initializer_list<Scalar>& value)
        {
            InitKernel();
            Resize(value.size());

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, std::data(value), CL_TRUE);
        }
        Vector::Vector(const eilig::Vector& input)
        {
            InitKernel();
            Resize(input.GetRows());

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, &input.data_[0], CL_TRUE);
        }
        Vector::Vector(NumberRows numberRows)
        {
            InitKernel();
            Resize(numberRows);
        }
        Vector::Vector(NumberRows numberRows, Scalar value)
        {
            InitKernel();
            Resize(numberRows, value);
        }
        Vector::Vector(Vector&& input) noexcept
        {
            (*this) = std::move(input);
        }
        
        eilig::Vector Vector::Convert() const
        {
			auto res = eilig::Vector(numberRows_);
            
            dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, &res.data_[0], CL_TRUE);

            return res;
        }
        void Vector::Resize(NumberRows numberRows)
        {
            club::Error error;
            club::Events events(1);
            Scalar zero{ 0.0 };

            if (numberRows == 0)
            {
                throw std::invalid_argument("Vector dimension cannot be zero.");
            }

            if (numberRows_ == numberRows)
            {
                return;
            }

            numberRows_ = numberRows;
            dataGPU_ = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_);

            error = clEnqueueFillBuffer(GetContext()->GetQueues()[deviceIndex_], dataGPU_->Get(), &zero, sizeof(Scalar), 0, sizeof(Scalar) * numberRows_, 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);
        }
        void Vector::Resize(NumberRows numberRows, Scalar value)
        {
            Resize(numberRows);
            (*this) = value;
        }
        void Vector::Fill(Scalar value)
        {
            (*this) = value;
        }
        
        EntryProxy Vector::operator()(Index row)
        {
            return EntryProxy(dataGPU_, row, deviceIndex_);
        }
   
        Vector& Vector::operator=(Scalar rhs)
        {
            club::Error error;
            Index globalSize[1];

			const auto& dimension = kernel_->kVectorCopyS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernel_->kVectorCopyS_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kVectorCopyS_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kVectorCopyS_->SetArg(2, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kVectorCopyS_->GetKernel(),
                kernel_->kVectorCopyS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return *this;
        }
        Vector& Vector::operator=(const Vector& rhs)
        {
			InitKernel();

            numberRows_ = rhs.numberRows_;
			deviceIndex_ = rhs.deviceIndex_;
            dataGPU_ = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_);

            clEnqueueCopyBuffer(GetContext()->GetQueues()[deviceIndex_], rhs.dataGPU_->Get(), dataGPU_->Get(), 0, 0, sizeof(Scalar) * numberRows_, 0, NULL, NULL);

            return *this;
        }
        Vector& Vector::operator=(Vector&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }

            kernel_ = rhs.kernel_;
			deviceIndex_ = rhs.deviceIndex_;
            numberRows_ = rhs.numberRows_;
            dataGPU_ = BufferPtr(std::move(rhs.dataGPU_));

            return *this;
        }
        Vector Vector::operator+(Scalar rhs) const
        {
            Vector res(*this);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kVectorAddS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kVectorAddS_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kVectorAddS_->SetArg(1, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kVectorAddS_->SetArg(2, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kVectorAddS_->GetKernel(),
                res.kernel_->kVectorAddS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector Vector::operator+(const Vector& rhs) const
        {
            Vector res(*this);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kVectorAddV_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kVectorAddV_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kVectorAddV_->SetArg(1, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kVectorAddV_->SetArg(2, sizeof(cl_mem), &rhs.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kVectorAddV_->GetKernel(),
                res.kernel_->kVectorAddV_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector Vector::operator+() const
        {
            return (*this);
        }
        Vector operator+(Scalar lhs, const Vector& rhs)
        {
            return rhs + lhs;
        }
        Vector Vector::operator-(Scalar rhs) const
        {
            Vector res(*this);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kVectorSubS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kVectorSubS_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kVectorSubS_->SetArg(1, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kVectorSubS_->SetArg(2, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kVectorSubS_->GetKernel(),
                res.kernel_->kVectorSubS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector Vector::operator-(const Vector& rhs) const
        {
            Vector res(*this);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kVectorSubV_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kVectorSubV_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kVectorSubV_->SetArg(1, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kVectorSubV_->SetArg(2, sizeof(cl_mem), &rhs.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kVectorSubV_->GetKernel(),
                res.kernel_->kVectorSubV_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector Vector::operator-() const
        {
            return -1.0 * (*this);
        }
        Vector operator-(Scalar lhs, const Vector& rhs)
        {
            return -rhs + lhs;
        }
        Vector Vector::operator*(Scalar rhs) const
        {
            Vector res(*this);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kVectorMulS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kVectorMulS_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kVectorMulS_->SetArg(1, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kVectorMulS_->SetArg(2, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kVectorMulS_->GetKernel(),
                res.kernel_->kVectorMulS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector operator*(Scalar lhs, const Vector& rhs)
        {
            return rhs * lhs;
        }
        
        Vector& Vector::SwapRows(Index row1, Index row2)
        {
            Scalar aux1;
            Scalar aux2;

            dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row1, sizeof(Scalar), &aux1, CL_TRUE);
            dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row2, sizeof(Scalar), &aux2, CL_TRUE);

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row1, sizeof(Scalar), &aux2, CL_TRUE);
            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row2, sizeof(Scalar), &aux1, CL_TRUE);

            return *this;
        }        
        Vector Vector::Region(Index row1, Index row2) const
        {
            Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index aux2 = row1 <= row2 ? row1 : row2;
            Vector res(aux1);
            Scalars data(aux1);

            dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * aux2, sizeof(Scalar) * aux1, &data[0], CL_TRUE);
            res.dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * aux1, &data[0], CL_TRUE);

            return res;
        }
        void Vector::Replace(Index row1, const Vector& in)
        {
            NumberRows numberRows = in.GetRows();
            Scalars data(numberRows);

            in.dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows, &data[0], CL_TRUE);
            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row1, sizeof(Scalar) * numberRows, &data[0], CL_TRUE);
        }
        void Vector::Replace(Index row1, const eilig::Vector& in)
        {
            NumberRows numberRows = in.GetRows();

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row1, sizeof(Scalar) * numberRows, &in.data_[0], CL_TRUE);
        }
        NumberRows Vector::GetRows() const
        {
            return numberRows_;
        }
        NumberCols Vector::GetCols() const
        {
            return 1;
        }
        Scalar Vector::GetValue(Index row) const
        {
            Scalar res{ 0.0 };

            dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * row, sizeof(Scalar), &res, CL_TRUE);

            return res;
        }
        BufferPtr Vector::GetDataGPU() const
        {
            return dataGPU_;
        }
        KernelVectorPtr Vector::GetKernel() const
        {
            return KernelVectorPtr();
        }
        const DeviceIndex& Vector::GetDeviceIndex() const
        {
            return deviceIndex_;
        }
       
        void Vector::Equal(Index row, Scalar value)
        {
            (*this)(row) = value;
        }
        void Vector::Equal(Scalar value)
        {
            (*this) = value;
        }
        void Vector::Equal(const Vector& value)
        {
            (*this) = value;
        }
        void Vector::Equal(const eilig::Vector& value)
        {
            Resize(value.GetRows());

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, &value.data_[0], CL_TRUE);
        }
        void Vector::Equal(const std::initializer_list<Scalar>& value)
        {
            Resize(value.size());

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, std::data(value), CL_TRUE);
        }
        
        void Vector::SetDevice(DeviceIndex deviceIndex)
        {
            if (deviceIndex >= GetContext()->GetDevices().size())
            {
                throw std::out_of_range("Device index is out of range.");
            }

			deviceIndex_ = deviceIndex;

            if (dataGPU_)
            {
			    clEnqueueMigrateMemObjects(GetContext()->GetQueues()[deviceIndex_], 1, &dataGPU_->Get(), 0, 0, NULL, NULL);
            }
        }
        void Vector::InitKernel()
        {
			kernel_ = CreateKernelVector();

            if(!kernel_)
            {
                throw std::runtime_error("Failed to create kernel.");
            }
        }
    }
} /* namespace eilig */