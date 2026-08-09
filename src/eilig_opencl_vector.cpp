#include "eilig_opencl_vector.hpp"

namespace eilig
{
    namespace opencl
    {
        Vector::Vector(const Vector& input)
        {
            (*this) = input;
        }
        Vector::Vector(KernelsPtr kernels)
        {
            SetKernels(kernels);
            Resize(1);
        }
        Vector::Vector(KernelsPtr kernels, const std::initializer_list<Scalar>& value)
        {
            SetKernels(kernels);
            Resize(value.size());

            dataGPU_->Write(0, sizeof(Scalar) * numberRows_, std::data(value), CL_TRUE);
        }
        Vector::Vector(KernelsPtr kernels, const eilig::Vector& input)
        {
            SetKernels(kernels);
            Resize(input.GetRows());

            dataGPU_->Write(0, sizeof(Scalar) * numberRows_, &input.data_[0], CL_TRUE);
        }
        Vector::Vector(KernelsPtr kernels, NumberRows numberRows)
        {
            SetKernels(kernels);
            Resize(numberRows);
        }
        Vector::Vector(KernelsPtr kernels, NumberRows numberRows, Scalar value)
        {
            SetKernels(kernels);
            Resize(numberRows, value);
        }
        Vector::Vector(Vector&& input) noexcept
        {
            (*this) = std::move(input);
        }
        eilig::Vector Vector::Convert() const
        {
			auto res = eilig::Vector(numberRows_);
            
            dataGPU_->Read(0, sizeof(Scalar) * numberRows_, &res.data_[0], CL_TRUE);

            return res;
        }
        void Vector::Resize(NumberRows numberRows)
        {
            club::Error error;
            club::Events events(1);
            Scalar value{ 0.0 };

            if (numberRows == 0)
            {
                throw std::invalid_argument("Vector dimensions cannot be zero.");
            }

            if (numberRows_ == numberRows)
            {
                return;
            }

            numberRows_ = numberRows;
            dataGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_);

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), dataGPU_->Get(), &value, sizeof(Scalar), 0, sizeof(Scalar) * numberRows_, 0, NULL, &events[0]);
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
            return EntryProxy(dataGPU_, row);
        }
        Vector& Vector::operator=(Scalar rhs)
        {
            Equal(rhs);

            return *this;
        }
        Vector& Vector::operator=(const Vector& rhs)
        {
            Equal(rhs);

            return *this;
        }
        Vector& Vector::operator=(Vector&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }

            kernels_ = rhs.kernels_;
            numberRows_ = rhs.numberRows_;
            dataGPU_ = BufferPtr(std::move(rhs.dataGPU_));

            return *this;
        }
        Vector Vector::operator+(Scalar rhs) const
        {
            Vector res(*this);

			res.Add(rhs);

            return res;
        }
        Vector Vector::operator+(const Vector& rhs) const
        {
            Vector res(*this);

            res.Add(rhs);

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

            res.Sub(rhs);

            return res;
        }
        Vector Vector::operator-(const Vector& rhs) const
        {
            Vector res(*this);

            res.Sub(rhs);

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

            res.Mul(rhs);

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

            dataGPU_->Read(sizeof(Scalar) * row1, sizeof(Scalar), &aux1, CL_TRUE);
            dataGPU_->Read(sizeof(Scalar) * row2, sizeof(Scalar), &aux2, CL_TRUE);

            dataGPU_->Write(sizeof(Scalar) * row1, sizeof(Scalar), &aux2, CL_TRUE);
            dataGPU_->Write(sizeof(Scalar) * row2, sizeof(Scalar), &aux1, CL_TRUE);

            return *this;
        }
        Vector Vector::Region(Index row1, Index row2) const
        {
            Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index aux2 = row1 <= row2 ? row1 : row2;
            Vector res(kernels_, aux1);
            Scalars data(aux1);

            dataGPU_->Read(sizeof(Scalar) * aux2, sizeof(Scalar) * aux1, &data[0], CL_TRUE);
            res.dataGPU_->Write(0, sizeof(Scalar) * aux1, &data[0], CL_TRUE);

            return res;
        }
        void Vector::Replace(Index row1, const Vector& in)
        {
            NumberRows numberRows = in.GetRows();
            Scalars data(numberRows);

            in.dataGPU_->Read(0, sizeof(Scalar) * numberRows, &data[0], CL_TRUE);
            dataGPU_->Write(sizeof(Scalar) * row1, sizeof(Scalar) * numberRows, &data[0], CL_TRUE);
        }
        void Vector::Replace(Index row1, const eilig::Vector& in)
        {
            NumberRows numberRows = in.GetRows();

            dataGPU_->Write(sizeof(Scalar) * row1, sizeof(Scalar) * numberRows, &in.data_[0], CL_TRUE);
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

            dataGPU_->Read(sizeof(Scalar) * row, sizeof(Scalar), &res, CL_TRUE);

            return res;
        }
        KernelsPtr Vector::GetKernels() const
        {
            return kernels_;
        }
        BufferPtr Vector::GetDataGPU() const
        {
            return dataGPU_;
        }
        void Vector::Equal(Index row, Scalar value)
        {
            (*this)(row) = value;
        }
        void Vector::Equal(Scalar value)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kVectorCopyS_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kVectorCopyS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kVectorCopyS_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorCopyS_->SetArg(2, sizeof(Scalar), &value);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kVectorCopyS_->GetKernel(),
                kernels_->kVectorCopyS_->GetDim(), NULL, globalSize,
                &kernels_->kVectorCopyS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }
        }
        void Vector::Equal(const Vector& value)
        {
            kernels_ = value.kernels_;
            numberRows_ = value.numberRows_;
            dataGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_);

            clEnqueueCopyBuffer(kernels_->context_->GetQueue(), value.dataGPU_->Get(), dataGPU_->Get(), 0, 0, sizeof(Scalar) * numberRows_, 0, NULL, NULL);
        }
        void Vector::Equal(const eilig::Vector& value)
        {
            Resize(value.GetRows());

            dataGPU_->Write(0, sizeof(Scalar) * numberRows_, &value.data_[0], CL_TRUE);
        }
        void Vector::Equal(const std::initializer_list<Scalar>& value)
        {
            Resize(value.size());

            dataGPU_->Write(0, sizeof(Scalar) * numberRows_, std::data(value), CL_TRUE);
        }
        void Vector::Add(Scalar value)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kVectorAddS_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kVectorAddS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kVectorAddS_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorAddS_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorAddS_->SetArg(3, sizeof(Scalar), &value);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kVectorAddS_->GetKernel(),
                kernels_->kVectorAddS_->GetDim(), NULL, globalSize,
                &kernels_->kVectorAddS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }
        }
        void Vector::Add(const Vector& value)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kVectorAddV_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kVectorAddV_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kVectorAddV_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorAddV_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorAddV_->SetArg(3, sizeof(cl_mem), &value.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kVectorAddV_->GetKernel(),
                kernels_->kVectorAddV_->GetDim(), NULL, globalSize,
                &kernels_->kVectorAddV_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }
        }
        void Vector::Sub(Scalar value)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kVectorSubS_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kVectorSubS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kVectorSubS_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorSubS_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorSubS_->SetArg(3, sizeof(Scalar), &value);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kVectorSubS_->GetKernel(),
                kernels_->kVectorSubS_->GetDim(), NULL, globalSize,
                &kernels_->kVectorSubS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }
        }
        void Vector::Sub(const Vector& value)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kVectorSubV_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kVectorSubV_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kVectorSubV_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorSubV_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorSubV_->SetArg(3, sizeof(cl_mem), &value.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kVectorSubV_->GetKernel(),
                kernels_->kVectorSubV_->GetDim(), NULL, globalSize,
                &kernels_->kVectorSubV_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }
        }
        void Vector::Mul(Scalar value)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kVectorMulS_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kVectorMulS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kVectorMulS_->SetArg(1, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorMulS_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kVectorMulS_->SetArg(3, sizeof(Scalar), &value);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kVectorMulS_->GetKernel(),
                kernels_->kVectorMulS_->GetDim(), NULL, globalSize,
                &kernels_->kVectorMulS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }
        }
        void Vector::SetKernels(KernelsPtr kernels)
        {
            if(kernels == nullptr)
            {
                throw std::invalid_argument("Kernels cannot be null.");
			}

			kernels_ = kernels;
        }
    }
} /* namespace eilig */