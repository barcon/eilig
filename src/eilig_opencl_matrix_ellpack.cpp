#ifdef EILIG_ENABLE_OPENCL

#include "eilig_routines.hpp"
#include "eilig_opencl_matrix_ellpack.hpp"
#include <algorithm>

namespace eilig
{
    namespace opencl
    {
        Ellpack::Ellpack(KernelsPtr kernels)
        {
            kernels_ = kernels;
            Resize(1, 1);
        }
        Ellpack::Ellpack(Ellpack&& input) noexcept
        {
            (*this) = std::move(input);
        }
        Ellpack::Ellpack(const Ellpack& input)
        {
            (*this) = input;
        }
        Ellpack::Ellpack(KernelsPtr kernels, const eilig::Matrix& input)
        {
            kernels_ = kernels;
            Resize(input.GetRows(), input.GetCols());

            for (Index i = 0; i < numberRows_; i++)
            {
                for (Index j = 0; j < numberCols_; j++)
                {
                    if (utils::math::IsAlmostEqual(input(i, j), 0.0, 5))
                    {
                        continue;
                    }

                    (*this)(i, j) = input(i, j);
                }
            }
        }
        Ellpack::Ellpack(KernelsPtr kernels, const eilig::Ellpack& input)
        {
            club::Error error;
            club::Events events(3);

            kernels_ = kernels;
            Resize(input.GetRows(), input.GetCols());
            Expand(input.GetWidth());

            error = clEnqueueWriteBuffer(kernels_->context_->GetQueue(), countGPU_->Get(), CL_FALSE, 0, sizeof(Index) * numberRows_, &input.count_[0], 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            error = clEnqueueWriteBuffer(kernels_->context_->GetQueue(), positionGPU_->Get(), CL_FALSE, 0, sizeof(Index) * numberRows_ * width_, &input.position_[0], 0, NULL, &events[1]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            error = clEnqueueWriteBuffer(kernels_->context_->GetQueue(), dataGPU_->Get(), CL_FALSE, 0, sizeof(Scalar) * numberRows_ * width_, &input.data_[0], 0, NULL, &events[2]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);
        }
        Ellpack::Ellpack(KernelsPtr kernels, NumberRows numberRows, NumberCols numberCols)
        {
            kernels_ = kernels;
            Resize(numberRows, numberCols);
        }
        Ellpack::Ellpack(KernelsPtr kernels, NumberRows numberRows, NumberCols numberCols, Type type)
        {            
            kernels_ = kernels;

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
        bool Ellpack::IsUsed(Index row, Index col) const
        {
            Index position{ 0 };

            return IsUsed(row, col, position);
        }
        bool Ellpack::IsUsed(Index row, Index col, Index& position) const
        {
            Index count;
            Indices buffer;

            buffer.resize(width_);

            countGPU_->Read(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);
            positionGPU_->Read(sizeof(Index) * row * width_, sizeof(Index) * width_, &buffer[0], CL_TRUE);

            for (Index k = 0; k < count; ++k)
            {
                if (col == buffer[k])
                {
                    position = k;
                    return true;
                }
                else if (buffer[k] > col)
                {
                    position = k;
                    return false;
                }
            }

            position = count;
            return false;
        }
        Index Ellpack::Add(Index row, Index col)
        {
            Index count{ 0 };
            Index position{ 0 };
            Scalar zeroScalar{ 0.0 };

            countGPU_->Read(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);

            if (!IsUsed(row, col, position))
            {
                if (position < count)
                {
                    if (count == width_)
                    {
                        Expand(width_ + GrowthRate());
                    }

                    ShiftRight(row, position);
                }
                else if (position == count)
                {
                    if (count == width_)
                    {
                        Expand(width_ + GrowthRate());
                    }
                }
                
                count += 1;
                countGPU_->Write(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);
                positionGPU_->Write(sizeof(Index) * (row * width_ + position), sizeof(Index), &col, CL_TRUE);
                dataGPU_->Write(sizeof(Scalar) * (row * width_ + position), sizeof(Scalar), &zeroScalar, CL_TRUE);
            }

            return (row * width_ + position);
        }
        void Ellpack::Remove(Index row, Index col)
        {
            Index count{ 0 };
            Index position{ 0 };
            Index zeroIndex{ 0 };
            Scalar zeroScalar{ 0.0 };

            countGPU_->Read(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);

            if (IsUsed(row, col, position))
            {
                ShiftLeft(row, position); 
                
                positionGPU_->Write(sizeof(Index) * (row * width_ + count - 1), sizeof(Index), &zeroIndex, CL_TRUE);
                dataGPU_->Write(sizeof(Scalar) * (row * width_ + count - 1), sizeof(Scalar), &zeroScalar, CL_TRUE);
                count -= 1;
                countGPU_->Write(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);
               
                Shrink();
            }
        }
        void Ellpack::Resize(Index numberRows, Index numberCols)
        {
            club::Error error;
            club::Events events(3);
            Index rhs{ 0 };
            Scalar zero{ 0 };
            Index expansion{ 0 };

            numberRows_ = numberRows;
            numberCols_ = numberCols;
            width_ = GrowthRate() > numberCols_ ? numberCols_ : GrowthRate();
            countGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_);
            dataGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_ * width_);
            positionGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_ * width_);

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), countGPU_->Get(), &rhs, sizeof(Index), 0, sizeof(Index) * numberRows_, 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel Resize: " + club::messages.at(error));
            }

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), positionGPU_->Get(), &rhs, sizeof(Index), 0, sizeof(Index) * numberRows_ * width_, 0, NULL, &events[1]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel Resize: " + club::messages.at(error));
            }

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), dataGPU_->Get(), &zero, sizeof(Scalar), 0, sizeof(Scalar) * numberRows_ * width_, 0, NULL, &events[2]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel Resize: " + club::messages.at(error));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);
        }
        void Ellpack::Resize(Index numberRows, Index numberCols, Scalar value)
        {
            Resize(numberRows, numberCols);
            Fill(value);
        }
        void Ellpack::Fill(Scalar value)
        {
            (*this) = value;
        }
        void Ellpack::Clear()
        {
            club::Error error;
            club::Events events(3);
            Index count{ 0 };
            Index position{ 0 };
            Scalar zero{ 0.0 };

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), countGPU_->Get(), &count, sizeof(Index), 0, sizeof(Index) * numberRows_, 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel Resize: " + club::messages.at(error));
            }

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), positionGPU_->Get(), &position, sizeof(Index), 0, sizeof(Index) * numberRows_ * width_, 0, NULL, &events[1]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel Resize: " + club::messages.at(error));
            }

            error = clEnqueueFillBuffer(kernels_->context_->GetQueue(), dataGPU_->Get(), &zero, sizeof(Scalar), 0, sizeof(Scalar) * numberRows_ * width_, 0, NULL, &events[2]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel Resize: " + club::messages.at(error));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

            Shrink();
        }
        void Ellpack::Init(eilig::Ellpack& input)
        {
            numberRows_ = input.numberRows_;
            numberCols_ = input.numberCols_;
            width_ = input.width_;

            countGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_);
            dataGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_ * width_);
            positionGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_ * width_);

            countGPU_->Write(0, sizeof(Index) * numberRows_, &input.count_[0]);
            dataGPU_->Write(0, sizeof(Scalar) * numberRows_ * width_, &input.data_[0]);
            positionGPU_->Write(0, sizeof(Index) * numberRows_ * width_, &input.position_[0], CL_TRUE);
        }
        void Ellpack::Dump() const
        {
            Scalars data{};
            Indices position{};
            Indices count{};

            logger::Info(headerEilig, "Matrix Ellpack CL (%zu x %zu):", numberRows_, numberCols_);

            std::cout << "Rows: " << numberRows_ << std::endl;
            std::cout << "Cols: " << numberCols_ << std::endl;
            std::cout << "Width: " << width_ << std::endl;
            std::cout << std::endl;

            count.resize(numberRows_);
            data.resize(numberRows_ * width_);
            position.resize(numberRows_ * width_);

            countGPU_->Read(0, sizeof(Index) * numberRows_, &count[0], CL_TRUE);
            dataGPU_->Read(0, sizeof(Scalar) * numberRows_ * width_, &data[0], CL_TRUE);
            positionGPU_->Read(0, sizeof(Index) * numberRows_ * width_, &position[0], CL_TRUE);

            std::cout << "Count: " << std::endl;
            for (Index i = 0; i < numberRows_; ++i)
            {
                std::cout << count[i] << std::endl;
            }
            std::cout << std::endl;

            std::cout << "Position: " << std::endl;
            for (Index i = 0; i < numberRows_; ++i)
            {
                for (Index j = 0; j < width_; ++j)
                {
                    std::cout << position[i * width_ + j] << "\t";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

            std::cout << "Data: " << std::endl;
            for (Index i = 0; i < numberRows_; ++i)
            {
                for (Index j = 0; j < width_; ++j)
                {
                    std::cout << utils::string::Format("%12.5g", data[i * width_ + j]) << "\t";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;

        }
        void Ellpack::Expand(NumberCols width)
        {
            Index expansion = width == 0 ? GrowthRate() : width;

            expansion = expansion > numberCols_ ? numberCols_ : expansion;
            
            if (expansion > width_)
            {
                BufferPtr position = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_ * expansion);
                BufferPtr data = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_ * expansion);

                club::Error error{ CL_SUCCESS };
                club::Events events(2);
                Index globalSize[2];

                const auto& localSize = kernels_->kEllpackExpandPosition_->GetLocalSize();

                globalSize[0] = GlobalSize(numberRows_, localSize[0]);
                globalSize[1] = GlobalSize(expansion, localSize[1]);

                kernels_->kEllpackExpandPosition_->SetArg(0, sizeof(Index), &numberRows_);
                kernels_->kEllpackExpandPosition_->SetArg(1, sizeof(Index), &width_);
                kernels_->kEllpackExpandPosition_->SetArg(2, sizeof(Index), &expansion);
                kernels_->kEllpackExpandPosition_->SetArg(3, sizeof(cl_mem), &position->Get());
                kernels_->kEllpackExpandPosition_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());

                error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                    kernels_->kEllpackExpandPosition_->GetKernel(),
                    kernels_->kEllpackExpandPosition_->GetDim(), NULL, globalSize,
                    &kernels_->kEllpackExpandPosition_->GetLocalSize()[0], 0, NULL, &events[0]);

                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                kernels_->kEllpackExpandData_->SetArg(0, sizeof(Index), &numberRows_);
                kernels_->kEllpackExpandData_->SetArg(1, sizeof(Index), &width_);
                kernels_->kEllpackExpandData_->SetArg(2, sizeof(Index), &expansion);
                kernels_->kEllpackExpandData_->SetArg(3, sizeof(cl_mem), &data->Get());
                kernels_->kEllpackExpandData_->SetArg(4, sizeof(cl_mem), &dataGPU_->Get());

                error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                    kernels_->kEllpackExpandData_->GetKernel(),
                    kernels_->kEllpackExpandData_->GetDim(), NULL, globalSize,
                    &kernels_->kEllpackExpandData_->GetLocalSize()[0], 0, NULL, &events[1]);

                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

                positionGPU_ = BufferPtr(std::move(position));
                dataGPU_ = BufferPtr(std::move(data));
                width_ = expansion;
            }
        }
        void Ellpack::Shrink()
        {
            Index shrinkage = std::max({ MaxCount(), GrowthRate() });

            shrinkage = shrinkage > numberCols_ ? numberCols_ : shrinkage;

            if (shrinkage < width_)
            {
                BufferPtr data = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_ * shrinkage);
                BufferPtr position = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_ * shrinkage);

                club::Error error;
                club::Events events(2);
                Index globalSize[2];

                const auto& localSize = kernels_->kEllpackShrinkPosition_->GetLocalSize();

                globalSize[0] = GlobalSize(numberRows_, localSize[0]);
                globalSize[1] = GlobalSize(shrinkage, localSize[1]);

                kernels_->kEllpackShrinkPosition_->SetArg(0, sizeof(Index), &numberRows_);
                kernels_->kEllpackShrinkPosition_->SetArg(1, sizeof(Index), &width_);
                kernels_->kEllpackShrinkPosition_->SetArg(2, sizeof(Index), &shrinkage);
                kernels_->kEllpackShrinkPosition_->SetArg(3, sizeof(cl_mem), &position->Get());
                kernels_->kEllpackShrinkPosition_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());

                error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                    kernels_->kEllpackShrinkPosition_->GetKernel(),
                    kernels_->kEllpackShrinkPosition_->GetDim(), NULL, globalSize,
                    &kernels_->kEllpackShrinkPosition_->GetLocalSize()[0], 0, NULL, &events[0]);

                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                kernels_->kEllpackShrinkData_->SetArg(0, sizeof(Index), &numberRows_);
                kernels_->kEllpackShrinkData_->SetArg(1, sizeof(Index), &width_);
                kernels_->kEllpackShrinkData_->SetArg(2, sizeof(Index), &shrinkage);
                kernels_->kEllpackShrinkData_->SetArg(3, sizeof(cl_mem), &data->Get());
                kernels_->kEllpackShrinkData_->SetArg(4, sizeof(cl_mem), &dataGPU_->Get());

                error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                    kernels_->kEllpackShrinkData_->GetKernel(),
                    kernels_->kEllpackShrinkData_->GetDim(), NULL, globalSize,
                    &kernels_->kEllpackShrinkData_->GetLocalSize()[0], 0, NULL, &events[1]);

                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

                positionGPU_ = BufferPtr(std::move(position));
                dataGPU_ = BufferPtr(std::move(data));
                width_ = shrinkage;
            }
        }
        Index Ellpack::GrowthRate()
        {
             return static_cast<Index>(std::max(5.0, std::ceil(0.05 * numberCols_)));
        }
        Index Ellpack::MaxCount()
        {
            club::Error error;
            Index globalSize[1];
            Index ngroups{ 0 };
            Index res{ 0 };
            Indices partial;

            const auto& localSize = kernels_->kEllpackMaxCount_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            ngroups = (numberRows_ % localSize[0]) > 0 ? (numberRows_ / localSize[0] + 1) : (numberRows_ / localSize[0]);

            BufferPtr partialGPU = club::CreateBuffer(kernels_->context_, sizeof(Index) * ngroups);

            kernels_->kEllpackMaxCount_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackMaxCount_->SetArg(1, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackMaxCount_->SetArg(2, sizeof(cl_mem), &partialGPU->Get());
            kernels_->kEllpackMaxCount_->SetArg(3, localSize[0] * sizeof(size_t), NULL);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackMaxCount_->GetKernel(),
                kernels_->kEllpackMaxCount_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackMaxCount_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            partial.resize(ngroups);
            partialGPU->Read(0, sizeof(Index) * ngroups, &partial[0], CL_TRUE);

            res = *std::max_element(partial.begin(), partial.end());

            return res;
        }
        void Ellpack::ShiftRight(Index row, Index position)
        {
            Index count{ 0 };

            countGPU_->Read(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);

            if (position < count)
            {
                club::Error error;
                club::Events events(2);

                auto buffer1 = club::CreateBuffer(kernels_->context_, sizeof(Index) * width_);
                auto buffer2 = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * width_);

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), positionGPU_->Get(), buffer1->Get(), sizeof(Index) * row * width_, 0, sizeof(Index) * width_, 0, NULL, &events[0]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel 1: " + club::messages.at(error));
                }

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), dataGPU_->Get(), buffer2->Get(), sizeof(Scalar) * row * width_, 0, sizeof(Scalar) * width_, 0, NULL, &events[1]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel 2: " + club::messages.at(error));
                }

                clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), buffer1->Get(), positionGPU_->Get(), sizeof(Index) * (position), sizeof(Index) * (row * width_ + position + 1), sizeof(Index) * (count - position), 0, NULL, &events[0]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel 3: " + club::messages.at(error));
                }

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), buffer2->Get(), dataGPU_->Get(), sizeof(Scalar) * (position), sizeof(Scalar) * (row * width_ + position + 1), sizeof(Scalar) * (count - position), 0, NULL, &events[1]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel 4: " + club::messages.at(error));
                }

                clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

            }
        }
        void Ellpack::ShiftLeft(Index row, Index position)
        {
            club::Error error;
            club::Events events(2);
            Index count{ 0 };
            Index zeroIndex{ 0 };
            Scalar zeroScalar{ 0.0 };

            countGPU_->Read(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);

            if (position < count)
            {
                auto buffer1 = club::CreateBuffer(kernels_->context_, sizeof(Index) * width_);
                auto buffer2 = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * width_);

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), positionGPU_->Get(), buffer1->Get(), sizeof(Index) * row * width_, 0, sizeof(Index) * width_, 0, NULL, &events[0]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), dataGPU_->Get(), buffer2->Get(), sizeof(Scalar) * row * width_, 0, sizeof(Scalar) * width_, 0, NULL, &events[1]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), buffer1->Get(), positionGPU_->Get(), sizeof(Index) * (position + 1), sizeof(Index) * (row * width_ + position), sizeof(Index) * (count - position - 1), 0, NULL, &events[0]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), buffer2->Get(), dataGPU_->Get(), sizeof(Scalar) * (position + 1), sizeof(Scalar) * (row * width_ + position), sizeof(Scalar) * (count - position - 1), 0, NULL, &events[1]);
                if (error != CL_SUCCESS)
                {
                    logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
                }

                clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);
            }

            positionGPU_->Write(sizeof(Index) * (row * width_ + count - 1), sizeof(Index), &zeroIndex, CL_TRUE);
            dataGPU_->Write(sizeof(Scalar) * (row * width_ + count - 1), sizeof(Scalar), &zeroScalar, CL_TRUE);
            count -= 1;
            countGPU_->Write(sizeof(Index) * row, sizeof(Index), &count, CL_TRUE);
        }
        EntryProxy Ellpack::operator()(Index row, Index col)
        {
            Index index;

            index = Add(row, col);
            return EntryProxy(dataGPU_, index);
        }
        Ellpack& Ellpack::operator=(Scalar rhs)
        {
            club::Error error;
            Index globalSize[2];

            const auto& localSize = kernels_->kEllpackCopyS_->GetLocalSize();

            if (utils::math::IsAlmostEqual(rhs, 0.0, 5))
            {
                Clear();
                Shrink();
                return *this;
            }

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            width_ = numberCols_;
            positionGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_ * width_);
            dataGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_ * width_);

            kernels_->kEllpackCopyS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackCopyS_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackCopyS_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackCopyS_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackCopyS_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackCopyS_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackCopyS_->SetArg(6, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackCopyS_->GetKernel(),
                kernels_->kEllpackCopyS_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackCopyS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return *this;
        }
        Ellpack& Ellpack::operator=(Ellpack&& rhs) noexcept
        {
            if (&rhs == this)
            {
                return *this;
            }
 
            numberRows_ = rhs.numberRows_;
            numberCols_ = rhs.numberCols_;
            width_ = rhs.width_;
            countGPU_ = BufferPtr(std::move(rhs.countGPU_));
            positionGPU_ = BufferPtr(std::move(rhs.positionGPU_));
            dataGPU_ = BufferPtr(std::move(rhs.dataGPU_));
            kernels_ = KernelsPtr(std::move(rhs.kernels_));

            return *this;
        }
        Ellpack& Ellpack::operator=(const Ellpack& rhs)
        {
            club::Error error;
            club::Events events(3);

            kernels_ = rhs.kernels_;
            numberRows_ = rhs.numberRows_;
            numberCols_ = rhs.numberCols_;
            width_ = rhs.width_;

            countGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_);
            positionGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberRows_ * width_);
            dataGPU_ = club::CreateBuffer(kernels_->context_, sizeof(Scalar) * numberRows_ * width_);

            error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), rhs.countGPU_->Get(), countGPU_->Get(), 0, 0, sizeof(Index) * numberRows_, 0, NULL, &events[0]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), rhs.positionGPU_->Get(), positionGPU_->Get(), 0, 0, sizeof(Index) * numberRows_ * width_, 0, NULL, &events[1]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            error = clEnqueueCopyBuffer(kernels_->context_->GetQueue(), rhs.dataGPU_->Get(), dataGPU_->Get(), 0, 0, sizeof(Scalar) * numberRows_ * width_, 0, NULL, &events[2]);
            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            clWaitForEvents(static_cast<cl_uint>(events.size()), &events[0]);

            return *this;
        }
        Ellpack Ellpack::operator+(Scalar rhs) const
        {
            club::Error error;
            Ellpack res(kernels_);
            Index globalSize[2];

            const auto& localSize = kernels_->kEllpackAddS_->GetLocalSize();
            
            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(width_, localSize[1]);

            if (utils::math::IsAlmostEqual(rhs, 0.0, 5))
            {
                return *this;
            }

            res.Resize(numberRows_, numberCols_, rhs);

            kernels_->kEllpackAddS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackAddS_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackAddS_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackAddS_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackAddS_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackAddS_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackAddS_->SetArg(6, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackAddS_->SetArg(7, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackAddS_->SetArg(8, sizeof(cl_mem), &res.dataGPU_->Get());
            kernels_->kEllpackAddS_->SetArg(9, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackAddS_->GetKernel(),
                kernels_->kEllpackAddS_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackAddS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::operator+(const Ellpack& rhs) const
        {
            Indices count;
            Indices position;
            Scalars data;
            Ellpack res(*this);

            count.resize(rhs.numberRows_);
            position.resize(rhs.width_);
            data.resize(rhs.width_);

            rhs.countGPU_->Read(0, sizeof(Index) * rhs.numberRows_, &count[0], CL_TRUE);

            for (Index i = 0; i < rhs.numberRows_; ++i)
            {
                rhs.positionGPU_->Read(i * rhs.width_ * sizeof(Index), rhs.width_ * sizeof(Index), &position[0], CL_TRUE);
                rhs.dataGPU_->Read(i * rhs.width_ * sizeof(Scalar), rhs.width_ * sizeof(Scalar), &data[0], CL_TRUE);

                for (Index j = 0; j < count[i]; ++j)
                {
                    auto col = position[j];
                    auto value = data[j];

                    res(i, col) += value;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator+() const
        {
            return (*this);
        }
        Ellpack operator+(Scalar lhs, const Ellpack& rhs)
        {
            return rhs + lhs;
        }
        Ellpack Ellpack::operator-(Scalar rhs) const
        {
            club::Error error;
            Ellpack res(kernels_);
            Index globalSize[2];

            const auto& localSize = kernels_->kEllpackSubS_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            if (utils::math::IsAlmostEqual(rhs, 0.0, 5))
            {
                return *this;
            }

            res.Resize(numberRows_, numberCols_, -rhs);

            kernels_->kEllpackSubS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackSubS_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackSubS_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackSubS_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackSubS_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackSubS_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackSubS_->SetArg(6, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackSubS_->SetArg(7, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackSubS_->SetArg(8, sizeof(cl_mem), &res.dataGPU_->Get());
            kernels_->kEllpackSubS_->SetArg(9, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackSubS_->GetKernel(),
                kernels_->kEllpackSubS_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackSubS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::operator-(const Ellpack& rhs) const
        {
            Indices count;
            Indices position;
            Scalars data;
            Ellpack res(*this);

            count.resize(rhs.numberRows_);
            position.resize(rhs.width_);
            data.resize(rhs.width_);

            rhs.countGPU_->Read(0, sizeof(Index) * rhs.numberRows_, &count[0], CL_TRUE);

            for (Index i = 0; i < res.numberRows_; ++i)
            {
                rhs.positionGPU_->Read((i * rhs.width_) * sizeof(Index), rhs.width_ * sizeof(Index), &position[0], CL_TRUE);
                rhs.dataGPU_->Read((i * rhs.width_) * sizeof(Scalar), rhs.width_ * sizeof(Scalar), &data[0], CL_TRUE);

                for (Index j = 0; j < count[i]; ++j)
                {
                    auto col = position[j];
                    auto value = data[j];

                    res(i, col) -= value;
                }
            }

            return res;
        }
        Ellpack Ellpack::operator-() const
        {
            return -1.0 * (*this);
        }
        Ellpack operator-(Scalar lhs, const Ellpack& rhs)
        {
            return -rhs + lhs;
        }
        Ellpack Ellpack::operator*(Scalar rhs) const
        {
            club::Error error;
            Ellpack res(*this);
            Index globalSize[2];

            const auto& localSize = kernels_->kEllpackMulS_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(width_, localSize[1]);

            kernels_->kEllpackMulS_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackMulS_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackMulS_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackMulS_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackMulS_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackMulS_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackMulS_->SetArg(6, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackMulS_->SetArg(7, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackMulS_->SetArg(8, sizeof(cl_mem), &res.dataGPU_->Get());
            kernels_->kEllpackMulS_->SetArg(9, sizeof(Scalar), &rhs);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackMulS_->GetKernel(),
                kernels_->kEllpackMulS_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackMulS_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }       
        Ellpack Ellpack::operator*(const Ellpack& rhs) const
        {
            //TODO: If localMem bigger than max. allowable memory, it will not work. 
            //Adjust kernel to check if global_id < numberRows_
            Ellpack transpose = rhs.Transpose();
            Ellpack res(kernels_, numberRows_, rhs.numberCols_);
            club::Error error;
            Index globalSize[1];
            Index localMem{ numberCols_ };

            const auto& localSize = kernels_->kEllpackMulM_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.Expand(transpose.GetWidth());

            kernels_->kEllpackMulM_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackMulM_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackMulM_->SetArg(2, sizeof(Index), &rhs.numberCols_);
            kernels_->kEllpackMulM_->SetArg(3, sizeof(Index), &width_);
            kernels_->kEllpackMulM_->SetArg(4, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(5, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(6, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(7, sizeof(Index), &transpose.width_);
            kernels_->kEllpackMulM_->SetArg(8, sizeof(cl_mem), &transpose.countGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(9, sizeof(cl_mem), &transpose.positionGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(10, sizeof(cl_mem), &transpose.dataGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(11, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(12, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(13, sizeof(cl_mem), &res.dataGPU_->Get());
            kernels_->kEllpackMulM_->SetArg(14, sizeof(Scalar) * localMem, NULL);

            error = clEnqueueNDRangeKernel(res.kernels_->context_->GetQueue(),
                res.kernels_->kEllpackMulM_->GetKernel(),
                res.kernels_->kEllpackMulM_->GetDim(), NULL, globalSize,
                &res.kernels_->kEllpackMulM_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Vector Ellpack::operator*(const Vector& rhs) const
        {
            Vector res(kernels_, numberRows_, 0.0);
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kEllpackMulV_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);;

            kernels_->kEllpackMulV_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackMulV_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackMulV_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackMulV_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackMulV_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackMulV_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackMulV_->SetArg(6, sizeof(cl_mem), &rhs.GetDataGPU()->Get());
            kernels_->kEllpackMulV_->SetArg(7, sizeof(cl_mem), &res.GetDataGPU()->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackMulV_->GetKernel(),
                kernels_->kEllpackMulV_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackMulV_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }


            return res;
        }
        Ellpack operator*(Scalar lhs, const Ellpack& rhs)
        {
            return rhs * lhs;
        }
        Ellpack& Ellpack::SwapRows(Index row1, Index row2)
        {
            club::Error error;
            Index globalSize[1];

            const auto& localSize = kernels_->kEllpackSwapRows_->GetLocalSize();

            globalSize[0] = GlobalSize(width_, localSize[0]);

            kernels_->kEllpackSwapRows_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackSwapRows_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackSwapRows_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackSwapRows_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackSwapRows_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackSwapRows_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackSwapRows_->SetArg(6, sizeof(Index), &row1);
            kernels_->kEllpackSwapRows_->SetArg(7, sizeof(Index), &row2);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackSwapRows_->GetKernel(),
                kernels_->kEllpackSwapRows_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackSwapRows_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return *this;
        }
        Ellpack& Ellpack::SwapCols(Index col1, Index col2)
        {
            club::Error error;
            Index globalSize[1]{ numberRows_ };

            const auto& localSize = kernels_->kEllpackSwapCols_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);;

            kernels_->kEllpackSwapCols_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackSwapCols_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackSwapCols_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackSwapCols_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackSwapCols_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackSwapCols_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackSwapCols_->SetArg(6, sizeof(Index), &col1);
            kernels_->kEllpackSwapCols_->SetArg(7, sizeof(Index), &col2);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackSwapCols_->GetKernel(),
                kernels_->kEllpackSwapCols_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackSwapCols_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return *this;
        }
        Index Ellpack::FindWidthTranspose() const
        {
            club::Error error;
            Index globalSize[2];
            Index res;
            Indices count;

            const auto& localSize = kernels_->kEllpackFindWidthTranspose_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);
            globalSize[1] = GlobalSize(numberCols_, localSize[1]);

            BufferPtr countGPU = club::CreateBuffer(kernels_->context_, sizeof(Index) * numberCols_);

            kernels_->kEllpackFindWidthTranspose_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackFindWidthTranspose_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackFindWidthTranspose_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackFindWidthTranspose_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackFindWidthTranspose_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackFindWidthTranspose_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackFindWidthTranspose_->SetArg(6, sizeof(cl_mem), &countGPU->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackFindWidthTranspose_->GetKernel(),
                kernels_->kEllpackFindWidthTranspose_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackFindWidthTranspose_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            count.resize(numberCols_);
            countGPU->Read(0, sizeof(Index) * numberCols_, &count[0], CL_TRUE);

            res = *std::max_element(count.begin(), count.end());

            return res;
        }
        Ellpack Ellpack::Transpose() const
        {
            club::Error error;
            Index globalSize[1];
            Ellpack res(kernels_, numberCols_, numberRows_);

            const auto& localSize = kernels_->kEllpackTranspose_->GetLocalSize();

            globalSize[0] = GlobalSize(numberCols_, localSize[0]);

            res.Expand(FindWidthTranspose());

            kernels_->kEllpackTranspose_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackTranspose_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackTranspose_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackTranspose_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackTranspose_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackTranspose_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackTranspose_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackTranspose_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackTranspose_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackTranspose_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackTranspose_->GetKernel(),
                kernels_->kEllpackTranspose_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackTranspose_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::Diagonal() const
        {
            club::Error error;
            Index globalSize[1];
            Ellpack res(kernels_, numberRows_, numberCols_);

            const auto& localSize = kernels_->kEllpackDiagonal_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            kernels_->kEllpackDiagonal_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackDiagonal_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackDiagonal_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackDiagonal_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackDiagonal_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackDiagonal_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackDiagonal_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackDiagonal_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackDiagonal_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackDiagonal_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackDiagonal_->GetKernel(),
                kernels_->kEllpackDiagonal_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackDiagonal_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::Lower(bool diag) const
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
        Ellpack Ellpack::LowerWithDiagonal() const
        {
            club::Error error;
            Index globalSize[1];
            Ellpack res(kernels_, numberRows_, numberCols_);

            const auto& localSize = kernels_->kEllpackLower1_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.Expand(width_);

            kernels_->kEllpackLower1_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackLower1_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackLower1_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackLower1_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackLower1_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackLower1_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackLower1_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackLower1_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackLower1_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackLower1_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackLower1_->GetKernel(),
                kernels_->kEllpackLower1_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackLower1_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::LowerWithoutDiagonal() const
        {
            club::Error error;
            Index globalSize[1];
            Ellpack res(kernels_, numberRows_, numberCols_);

            const auto& localSize = kernels_->kEllpackLower2_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.Expand(width_);

            kernels_->kEllpackLower2_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackLower2_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackLower2_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackLower2_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackLower2_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackLower2_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackLower2_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackLower2_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackLower2_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackLower2_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackLower2_->GetKernel(),
                kernels_->kEllpackLower2_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackLower2_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::Upper(bool diag) const
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
        Ellpack Ellpack::UpperWithDiagonal() const
        {
            club::Error error;
            Index globalSize[1]{ numberRows_ };
            Ellpack res(kernels_, numberRows_, numberCols_);

            const auto& localSize = kernels_->kEllpackUpper1_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.Expand(width_);

            kernels_->kEllpackUpper1_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackUpper1_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackUpper1_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackUpper1_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackUpper1_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackUpper1_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackUpper1_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackUpper1_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackUpper1_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackUpper1_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackUpper1_->GetKernel(),
                kernels_->kEllpackUpper1_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackUpper1_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::UpperWithoutDiagonal() const
        {
            club::Error error;
            Index globalSize[1];
            Ellpack res(kernels_, numberRows_, numberCols_);

            const auto& localSize = kernels_->kEllpackUpper2_->GetLocalSize();

            globalSize[0] = GlobalSize(numberRows_, localSize[0]);

            res.Expand(width_);

            kernels_->kEllpackUpper2_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackUpper2_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackUpper2_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackUpper2_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackUpper2_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackUpper2_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackUpper2_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackUpper2_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackUpper2_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackUpper2_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackUpper2_->GetKernel(),
                kernels_->kEllpackUpper2_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackUpper2_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        Ellpack Ellpack::Region(Index row1, Index col1, Index row2, Index col2) const
        {
            Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
            Index aux3;
            Index aux4;

            club::Error error;
            Index globalSize[1]{ aux1 };
            Ellpack res(kernels_, aux1, aux2);

            const auto& localSize = kernels_->kEllpackRegion_->GetLocalSize();

            globalSize[0] = GlobalSize(aux1, localSize[0]);

            res.Expand(width_);

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

            kernels_->kEllpackRegion_->SetArg(0, sizeof(Index), &numberRows_);
            kernels_->kEllpackRegion_->SetArg(1, sizeof(Index), &numberCols_);
            kernels_->kEllpackRegion_->SetArg(2, sizeof(Index), &width_);
            kernels_->kEllpackRegion_->SetArg(3, sizeof(cl_mem), &countGPU_->Get());
            kernels_->kEllpackRegion_->SetArg(4, sizeof(cl_mem), &positionGPU_->Get());
            kernels_->kEllpackRegion_->SetArg(5, sizeof(cl_mem), &dataGPU_->Get());
            kernels_->kEllpackRegion_->SetArg(6, sizeof(Index), &res.width_);
            kernels_->kEllpackRegion_->SetArg(7, sizeof(cl_mem), &res.countGPU_->Get());
            kernels_->kEllpackRegion_->SetArg(8, sizeof(cl_mem), &res.positionGPU_->Get());
            kernels_->kEllpackRegion_->SetArg(9, sizeof(cl_mem), &res.dataGPU_->Get());
            kernels_->kEllpackRegion_->SetArg(10, sizeof(Index), &aux1);
            kernels_->kEllpackRegion_->SetArg(11, sizeof(Index), &aux2);
            kernels_->kEllpackRegion_->SetArg(12, sizeof(Index), &aux3);
            kernels_->kEllpackRegion_->SetArg(13, sizeof(Index), &aux4);

            error = clEnqueueNDRangeKernel(kernels_->context_->GetQueue(),
                kernels_->kEllpackRegion_->GetKernel(),
                kernels_->kEllpackRegion_->GetDim(), NULL, globalSize,
                &kernels_->kEllpackRegion_->GetLocalSize()[0], 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
            }

            return res;
        }
        void   Ellpack::Region(Index row1, Index col1, Index row2, Index col2, const Ellpack& in)
        {
            Index aux1 = row1 <= row2 ? (row2 - row1) + 1 : (row1 - row2) + 1;
            Index aux2 = col1 <= col2 ? (col2 - col1) + 1 : (col1 - col2) + 1;
            Index aux3;
            Index aux4;

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

            for (Index i = 0; i < aux1; ++i)
            {
                for (Index j = 0; j < aux2; ++j)
                {
                    (*this)(aux3 + i, aux4 + j) = in.GetValue(i, j);
                }
            }
        }
        NumberRows Ellpack::GetRows() const
        {
            return numberRows_;
        }
        NumberCols Ellpack::GetCols() const
        {
            return numberCols_;
        }
        NumberCols Ellpack::GetWidth() const
        {
            return width_;
        }
        Scalar Ellpack::GetValue(Index row, Index col) const
        {
            Scalar res{ 0.0 };
            Index position;
            Index index;

            if (IsUsed(row, col, position))
            {
                index = row * width_ + position;

                dataGPU_->Read(sizeof(Scalar) * index, sizeof(Scalar), &res, CL_TRUE);
            }

            return res;
        }
        KernelsPtr Ellpack::GetKernels() const
        {
            return kernels_;
        }
        BufferPtr Ellpack::GetCountGPU() const
        {
            return countGPU_;
        }
        BufferPtr Ellpack::GetPositionGPU() const
        {
            return positionGPU_;
        }
        BufferPtr Ellpack::GetDataGPU() const
        {
            return dataGPU_;
        }
        void Ellpack::SetValue(Index row, Index col, Scalar value)
        {
            if (utils::math::IsAlmostEqual(value, 0.0, 5))
            {
                Remove(row, col);
            }
            else
            {
                (*this)(row, col) = value;
            }
        }
    }
} /* namespace eilig */

#endif

/*

        void Add(Ellpack& out, const Ellpack& in, Scalar value)
        {
            out = in + value;
        }
        void Add(Ellpack& out, const Ellpack& in, const Ellpack& value)
        {
            out = in + value;
        }
        void Sub(Ellpack& out, const Ellpack& in, Scalar value)
        {
            out = in - value;
        }
        void Sub(Ellpack& out, const Ellpack& in, const Ellpack& value)
        {
            out = in - value;
        }
        void Mul(Ellpack& out, const Ellpack& in, Scalar value)
        {
            out = in * value;
        }
        void Mul(Ellpack& out, const Ellpack& in, const Ellpack& value)
        {
            out = in * value;
        }
        void Mul(Vector& out, const Ellpack& in, const Vector& value)
        {
            out = in * value;
        }

*/