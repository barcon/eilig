#include "eilig_opencl_matrix.hpp"

namespace eilig
{
    namespace opencl
    {
        Matrix::Matrix()
        {
            InitKernel();
            Resize(1, 1);
        }
        Matrix::Matrix(const Matrix& input)
        {
            (*this) = input;
        }
        Matrix::Matrix(const std::initializer_list<std::initializer_list<Scalar>>& value)
        {
            InitKernel();
            Resize(value.size(), value.begin()->size());

            Index i = 0;
            for (auto& outerItens : value)
            {
                if (outerItens.size() != numberCols_)
                {
                    throw std::invalid_argument("All rows must have the same number of columns.");
                }

                Index j = 0;
                for (auto& it : outerItens)
                {
                    (*this)(i, j) = it;

                    ++j;
                }

                ++i;
            }
        }
        Matrix::Matrix(const eilig::Matrix& input)
        {            
            InitKernel();            
            Resize(input.GetRows(), input.GetCols());

            dataGPU_->Write(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_ * numberCols_, &input.GetData()[0], CL_TRUE);
        }
        Matrix::Matrix(const eilig::Ellpack& input)
        {
			InitKernel();
            Resize(input.GetRows(), input.GetCols());

            const auto& count = input.GetCount();
            const auto& position = input.GetPosition();
            const auto& data = input.GetData();
            const auto& width = input.GetWidth();

            for (Index i = 0; i < numberRows_; ++i)
            {
                for (Index k = 0; k < count[i]; ++k)
                {
                    auto j = position[i * width + k];
                    (*this)(i, j) = data[i * width + k];
                }
            }
        }
        Matrix::Matrix(NumberRows numberRows, NumberCols numberCols)
        {
            InitKernel();
            Resize(numberRows, numberCols);
        }
        Matrix::Matrix(NumberRows numberRows, NumberCols numberCols, Type type)
        {
            InitKernel();

            switch (type)
            {
            case matrix_ones:
                Resize(numberRows, numberCols, 1.0);
                break;
            case matrix_diagonal:
                Resize(numberRows, numberCols, 0.0);

                for (Index i = 0; (i < numberRows) && (i < numberCols); ++i)
                {
                    (*this)(i, i) = 1.0;
                }
                break;
            case matrix_zeros:
            default:
                Resize(numberRows, numberCols, 0.0);
            }
        }
        Matrix::Matrix(Matrix&& input) noexcept
        {
            (*this) = std::move(input);
        }
    
        void Matrix::Resize(Index numberRows, Index numberCols)
        {
            club::Error error;
            club::Events events(1);
            Scalar zero{ 0 };

            if (numberRows == 0 || numberCols == 0)
            {
                throw std::invalid_argument("Matrix dimensions cannot be zero.");
            }

            if (numberRows == numberRows_ && numberCols == numberCols_)
            {
                return;
            }

            numberRows_ = numberRows;
            numberCols_ = numberCols;

            dataGPU_ = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_ * numberCols_);

            error = clEnqueueFillBuffer(GetContext()->GetQueues()[deviceIndex_], dataGPU_->Get(), &zero, sizeof(Scalar), 0, sizeof(Scalar) * numberRows_ * numberCols_, 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel FillBuffer 3: {}", club::messages.at(error)));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);
        }
        void Matrix::Resize(Index numberRows, Index numberCols, Scalar value)
        {
            Resize(numberRows, numberCols);
            (*this) = value;
        }
        void Matrix::Fill(Scalar value)
        {
            (*this) = value;
        }

        EntryProxy Matrix::operator()(Index row, Index col)
        {
            return EntryProxy(dataGPU_, row * numberCols_ + col, deviceIndex_);
        }
        
        Matrix& Matrix::operator=(Scalar rhs)
        {
            club::Error error;
            Index globalSize[2];

            const auto& dimension = kernel_->kMatrixCopyS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            dataGPU_ = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_ * numberCols_);

            kernel_->kMatrixCopyS_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixCopyS_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixCopyS_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixCopyS_->SetArg(3, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixCopyS_->GetKernel(),
                kernel_->kMatrixCopyS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return *this;
        }
        Matrix& Matrix::operator=(const Matrix& rhs)
        {
            club::Error error;
            club::Events events(1);

			InitKernel();

            numberRows_ = rhs.numberRows_;
            numberCols_ = rhs.numberCols_;
            deviceIndex_ = rhs.deviceIndex_;
            dataGPU_ = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_ * numberCols_);

            error = clEnqueueCopyBuffer(GetContext()->GetQueues()[deviceIndex_], rhs.dataGPU_->Get(), dataGPU_->Get(), 0, 0, sizeof(Scalar) * numberRows_ * numberCols_, 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

            return *this;
        }
        Matrix& Matrix::operator=(Matrix&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }

            kernel_ = rhs.kernel_;
            deviceIndex_ = rhs.deviceIndex_;
            numberRows_ = rhs.numberRows_;
            numberCols_ = rhs.numberCols_;
            dataGPU_ = BufferPtr(std::move(rhs.dataGPU_));

            return *this;
        }
        Matrix Matrix::operator+(Scalar rhs) const
        {
            Matrix res(*this);
            club::Error error;
            Index globalSize[2];            

            const auto& dimension = res.kernel_->kMatrixAddS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            res.kernel_->kMatrixAddS_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kMatrixAddS_->SetArg(1, sizeof(Index), &res.numberCols_);
            res.kernel_->kMatrixAddS_->SetArg(2, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kMatrixAddS_->SetArg(3, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixAddS_->GetKernel(),
                res.kernel_->kMatrixAddS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::operator+(const Matrix& rhs) const
        {
            Matrix res(*this);
            club::Error error;
            Index globalSize[2];

            const auto& dimension = res.kernel_->kMatrixAddM_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            res.kernel_->kMatrixAddM_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kMatrixAddM_->SetArg(1, sizeof(Index), &res.numberCols_);
            res.kernel_->kMatrixAddM_->SetArg(2, sizeof(cl_mem), &rhs.dataGPU_->Get());
            res.kernel_->kMatrixAddM_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixAddM_->GetKernel(),
                res.kernel_->kMatrixAddM_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::operator+() const
        {
            return (*this);
        }
        Matrix operator+(Scalar lhs, const Matrix& rhs)
        {
            return rhs + lhs;
        }
        Matrix Matrix::operator-(Scalar rhs) const
        {
            Matrix res(*this);
            club::Error error;
            Index globalSize[2];

            const auto& dimension = res.kernel_->kMatrixSubS_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            res.kernel_->kMatrixSubS_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kMatrixSubS_->SetArg(1, sizeof(Index), &res.numberCols_);
            res.kernel_->kMatrixSubS_->SetArg(2, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kMatrixSubS_->SetArg(3, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixSubS_->GetKernel(),
                res.kernel_->kMatrixSubS_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::operator-(const Matrix& rhs) const
        {
            Matrix res(*this);
            club::Error error;
            Index globalSize[2];

            const auto& dimension = res.kernel_->kMatrixSubM_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            res.kernel_->kMatrixSubM_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kMatrixSubM_->SetArg(1, sizeof(Index), &res.numberCols_);
            res.kernel_->kMatrixSubM_->SetArg(2, sizeof(cl_mem), &rhs.dataGPU_->Get());
            res.kernel_->kMatrixSubM_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixSubM_->GetKernel(),
                res.kernel_->kMatrixSubM_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::operator-() const
        {
            return -1.0 * (*this);
        }
        Matrix operator-(Scalar lhs, const Matrix& rhs)
        {
            return -rhs + lhs;
        }
        Matrix Matrix::operator*(Scalar rhs) const
        {
            Matrix res(*this);
            club::Error error;
            Index globalSize[2];

            const auto& dimension = res.kernel_->kMatrixMulScalar_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            res.kernel_->kMatrixMulScalar_->SetArg(0, sizeof(Index), &res.numberRows_);
            res.kernel_->kMatrixMulScalar_->SetArg(1, sizeof(Index), &res.numberCols_);
            res.kernel_->kMatrixMulScalar_->SetArg(2, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kMatrixMulScalar_->SetArg(3, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixMulScalar_->GetKernel(),
                res.kernel_->kMatrixMulScalar_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::operator*(const Matrix& rhs) const
        {
            //TODO: If localMem bigger than max. allowable memory, it will not work. 
            //Adjust kernel to check if global_id < numberRows_

            Matrix res(numberRows_, rhs.numberCols_);
            Matrix transpose = rhs.Transpose();
            club::Error error;
            Index globalSize[1];
            Index localMem{ numberCols_ };

            const auto& dimension = res.kernel_->kMatrixMulMatrix_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixMulMatrix_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixMulMatrix_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixMulMatrix_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixMulMatrix_->SetArg(3, sizeof(Index), &transpose.numberRows_);
            res.kernel_->kMatrixMulMatrix_->SetArg(4, sizeof(Index), &transpose.numberCols_);
            res.kernel_->kMatrixMulMatrix_->SetArg(5, sizeof(cl_mem), &transpose.dataGPU_->Get());
            res.kernel_->kMatrixMulMatrix_->SetArg(6, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kMatrixMulMatrix_->SetArg(7, sizeof(Scalar) * localMem, NULL);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixMulMatrix_->GetKernel(),
                res.kernel_->kMatrixMulMatrix_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector Matrix::operator*(const Vector& rhs) const
        {
            Vector res(numberRows_);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = kernel_->kMatrixMulVector_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);;

            kernel_->kMatrixMulVector_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixMulVector_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixMulVector_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixMulVector_->SetArg(3, sizeof(cl_mem), &rhs.GetDataGPU()->Get());
            kernel_->kMatrixMulVector_->SetArg(4, sizeof(cl_mem), &res.GetDataGPU()->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixMulVector_->GetKernel(),
                kernel_->kMatrixMulVector_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix operator*(Scalar lhs, const Matrix& rhs)
        {
            return rhs * lhs;
        }

        Matrix& Matrix::SwapRows(Index row1, Index row2)
        {
            club::Error error;
            Index globalSize[1];

            const auto& dimension = kernel_->kMatrixSwapRows_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberCols_, localSize[0]);

            kernel_->kMatrixSwapRows_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixSwapRows_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixSwapRows_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixSwapRows_->SetArg(3, sizeof(Index), &row1);
            kernel_->kMatrixSwapRows_->SetArg(4, sizeof(Index), &row2);
            
            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixSwapRows_->GetKernel(),
                kernel_->kMatrixSwapRows_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return *this;
        }
        Matrix& Matrix::SwapCols(Index col1, Index col2)
        {
            club::Error error;
            Index globalSize[1]{ numberRows_ };

            const auto& dimension = kernel_->kMatrixSwapCols_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);;

            kernel_->kMatrixSwapCols_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixSwapCols_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixSwapCols_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixSwapCols_->SetArg(3, sizeof(Index), &col1);
            kernel_->kMatrixSwapCols_->SetArg(4, sizeof(Index), &col2);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixSwapCols_->GetKernel(),
                kernel_->kMatrixSwapCols_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return *this;
        }
        Scalar Matrix::Trace() const
        {
            club::Error error;
            Index globalSize[1];
            Scalars partial;
            Scalar res{ 0.0 };

            const auto& dimension = kernel_->kMatrixTrace_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            BufferPtr partialGPU = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_);

            kernel_->kMatrixTrace_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixTrace_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixTrace_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixTrace_->SetArg(3, sizeof(cl_mem), &partialGPU->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixTrace_->GetKernel(),
                kernel_->kMatrixTrace_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            partial.resize(numberRows_);
            partialGPU->Read(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, &partial[0], CL_TRUE);
         
            for (Index i = 0; i < partial.size(); i++)
            {
                res += partial[i];
            }

            return res;
        }
        Scalar Matrix::Sum() const
        {
            club::Error error;
            Index globalSize[1];
            Scalars partial;
            Scalar res{ 0.0 };

            const auto& dimension = kernel_->kMatrixSum_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            BufferPtr partialGPU = club::CreateBuffer(GetContext(), sizeof(Scalar) * numberRows_);

            kernel_->kMatrixSum_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixSum_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixSum_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixSum_->SetArg(3, sizeof(cl_mem), &partialGPU->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixSum_->GetKernel(),
                kernel_->kMatrixSum_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            partial.resize(numberRows_);
            partialGPU->Read(GetContext()->GetQueues()[deviceIndex_], 0, sizeof(Scalar) * numberRows_, &partial[0], CL_TRUE);

            for (Index i = 0; i < partial.size(); i++)
            {
                res += partial[i];
            }

            return res;
        }
        Matrix Matrix::Transpose() const
        {
            Matrix res(numberCols_, numberRows_);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kMatrixTranspose_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberCols_, localSize[0]);

            res.kernel_->kMatrixTranspose_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixTranspose_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixTranspose_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixTranspose_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixTranspose_->GetKernel(),
                res.kernel_->kMatrixTranspose_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::Diagonal() const
        {
            Matrix res(numberRows_, numberCols_, matrix_zeros);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kMatrixDiagonal_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixDiagonal_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixDiagonal_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixDiagonal_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixDiagonal_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixDiagonal_->GetKernel(),
                res.kernel_->kMatrixDiagonal_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::DiagonalScale(Scalar factor) const
        {
            Matrix res(*this);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kMatrixDiagonalScale_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixDiagonalScale_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixDiagonalScale_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixDiagonalScale_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixDiagonalScale_->SetArg(3, sizeof(Scalar), &factor);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixDiagonalScale_->GetKernel(),
                res.kernel_->kMatrixDiagonalScale_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Vector Matrix::DiagonalVector() const
        {
            club::Error error;
            Index globalSize[1];
            Vector res(std::min(numberRows_, numberCols_), 0.0);

            const auto& dimension = kernel_->kMatrixDiagonalVector_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(std::min(numberRows_, numberCols_), localSize[0]);

            kernel_->kMatrixDiagonalVector_->SetArg(0, sizeof(Index), &numberRows_);
            kernel_->kMatrixDiagonalVector_->SetArg(1, sizeof(Index), &numberCols_);
            kernel_->kMatrixDiagonalVector_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            kernel_->kMatrixDiagonalVector_->SetArg(3, sizeof(cl_mem), &res.GetDataGPU()->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                kernel_->kMatrixDiagonalVector_->GetKernel(),
                kernel_->kMatrixDiagonalVector_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::Lower(bool diag) const
        {
            if (diag)
            {
                return LowerWithDiagonal();
            }
            else
            {
                return LowerWithoutDiagonal();
            }
        }
        Matrix Matrix::LowerWithDiagonal() const
        {
            Matrix res(numberRows_, numberCols_);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kMatrixLower1_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixLower1_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixLower1_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixLower1_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixLower1_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixLower1_->GetKernel(),
                res.kernel_->kMatrixLower1_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::LowerWithoutDiagonal() const
        {
            Matrix res(numberRows_, numberCols_);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kMatrixLower2_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixLower2_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixLower2_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixLower2_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixLower2_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixLower2_->GetKernel(),
                res.kernel_->kMatrixLower2_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::Upper(bool diag) const
        {
            if (diag)
            {
                return UpperWithDiagonal();
            }
            else
            {
                return UpperWithoutDiagonal();
            }
        }
        Matrix Matrix::UpperWithDiagonal() const
        {
            club::Error error;
            Index globalSize[1]{ numberRows_ };
            Matrix res(numberRows_, numberCols_);

            const auto& dimension = res.kernel_->kMatrixUpper1_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixUpper1_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixUpper1_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixUpper1_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixUpper1_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixUpper1_->GetKernel(),
                res.kernel_->kMatrixUpper1_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::UpperWithoutDiagonal() const
        {
            Matrix res(numberRows_, numberCols_);
            club::Error error;
            Index globalSize[1];

            const auto& dimension = res.kernel_->kMatrixUpper2_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.kernel_->kMatrixUpper2_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixUpper2_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixUpper2_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixUpper2_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixUpper2_->GetKernel(),
                res.kernel_->kMatrixUpper2_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        Matrix Matrix::Region(Index row1, Index col1, Index row2, Index col2) const
        {
            Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
            Index aux3;
            Index aux4;

            Matrix res(aux1, aux2);
            club::Error error;
            Index globalSize[1]{ aux1 };

            const auto& dimension = kernel_->kMatrixRegion_->GetDim();
            const auto& localSize = GetContext()->GetLocalSize(deviceIndex_, dimension);

            globalSize[0] = GlobalSize(aux1, localSize[0]);

            if ((row1 <= row2) && (col1 <= col2))
            {
                aux3 = row1;
                aux4 = col1;
            }
            else if ((row1 >= row2) && (col1 <= col2))
            {
                aux3 = row2;
                aux4 = col1;
            }
            else if ((row1 >= row2) && (col1 >= col2))
            {
                aux3 = row2;
                aux4 = col2;
            }
            else
            {
                aux3 = row1;
                aux4 = col2;
            }

            res.kernel_->kMatrixRegion_->SetArg(0, sizeof(Index), &numberRows_);
            res.kernel_->kMatrixRegion_->SetArg(1, sizeof(Index), &numberCols_);
            res.kernel_->kMatrixRegion_->SetArg(2, sizeof(cl_mem), &dataGPU_->Get());
            res.kernel_->kMatrixRegion_->SetArg(3, sizeof(cl_mem), &res.dataGPU_->Get());
            res.kernel_->kMatrixRegion_->SetArg(4, sizeof(Index), &aux1);
            res.kernel_->kMatrixRegion_->SetArg(5, sizeof(Index), &aux2);
            res.kernel_->kMatrixRegion_->SetArg(6, sizeof(Index), &aux3);
            res.kernel_->kMatrixRegion_->SetArg(7, sizeof(Index), &aux4);

            error = clEnqueueNDRangeKernel(GetContext()->GetQueues()[deviceIndex_],
                res.kernel_->kMatrixRegion_->GetKernel(),
                res.kernel_->kMatrixRegion_->GetDim(), NULL, globalSize,
                &localSize[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
            }

            return res;
        }
        void Matrix::Replace(Index row1, Index col1, const Matrix& in)
        {
            NumberRows numberRows = in.GetRows();
            NumberCols numberCols = in.GetCols();

            for (Index i = 0; i < numberRows; ++i)
            {
                for (Index j = 0; j < numberCols; ++j)
                {
                    (*this)(row1 + i, col1 + j) = in.GetValue(i, j);
                }
            }
        }

        NumberRows Matrix::GetRows() const
        {
            return numberRows_;
        }
        NumberCols Matrix::GetCols() const
        {
            return numberCols_;
        }
        Scalar Matrix::GetValue(Index row, Index col) const
        {
            Scalar res{ 0.0 };

            dataGPU_->Read(GetContext()->GetQueues()[deviceIndex_], sizeof(Scalar) * (row * numberCols_ + col), sizeof(Scalar), &res, CL_TRUE);

            return res;
        }
        BufferPtr Matrix::GetDataGPU() const
        {
            return dataGPU_;
        }
        KernelMatrixPtr Matrix::GetKernel() const
        {
            return KernelMatrixPtr();
        }
        const DeviceIndex& Matrix::GetDeviceIndex() const
        {
            return deviceIndex_;
        }

        void Matrix::Equal(Index row, Index col, Scalar value)
        {
            (*this)(row, col) = value;
        }
        void Matrix::Equal(const Matrix& value)
        {
            (*this) = value;
        }
        void Matrix::Equal(const std::initializer_list<std::initializer_list<Scalar>>& value)
        {
            Resize(value.size(), value.begin()->size());

            Index i = 0;
            for (auto& outerItens : value)
            {
                if (outerItens.size() != numberCols_)
                {
                    throw std::invalid_argument("All rows must have the same number of columns.");
                }

                Index j = 0;
                for (auto& it : outerItens)
                {
                    (*this)(i, j) = it;
                    ++j;
                }

                ++i;
            }
        }

        void Matrix::SetDevice(DeviceIndex deviceIndex)
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
        void Matrix::InitKernel()
        {
            kernel_ = CreateKernelMatrix();

            if (!kernel_)
            {
                throw std::runtime_error("Failed to create kernel.");
            }
        }
    }
} /* namespace eilig */