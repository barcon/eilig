#include "eilig_routines.hpp"
#include "eilig_status.hpp"

#include <cmath>

namespace eilig
{
    Indices CreateIndices()
    {
        return Indices();
    }
    
    Scalar NormMax(const Vector& in)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        for (Index i = 0; i < in.GetRows(); ++i)
        {
            norm = std::abs(in(i));
            if (norm > res)
            {
                res = norm;
            }
        }
        return res;
    }
    Scalar NormP(const Vector& in, Scalar p)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        for (Index i = 0; i < in.GetRows(); ++i)
        {
            norm += std::pow(std::abs(in(i)), p);
        }

        res = std::pow(norm, 1. / p);
        return res;
    }
    Scalar NormP(const Matrix& in, Scalar p)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        const auto& data = in.GetData();

        for (Index i = 0; i < data.size(); ++i)
        {
            norm += std::pow(std::abs(data[i]), p);
        }

        res = std::pow(norm, 1. / p);
        return res;
    }
    Scalar NormP(const Ellpack& in, Scalar p)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        const auto& count = in.GetCount();
        const auto& data = in.GetData();
        const auto& width = in.GetWidth();

        for (Index i = 0; i < in.GetRows(); ++i)
        {
            for (Index j = 0; j < count[i]; ++j)
            {
                norm += std::pow(std::abs(data[i * width + j]), p);
            }
        }

        res = std::pow(norm, 1. / p);
        return res;
    }
    Scalar NormP2(const Vector& in)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        for (Index i = 0; i < in.GetRows(); ++i)
        {
            norm += in(i) * in(i);
        }

        res = std::sqrt(norm);
        return res;
    }
    Scalar NormP2(const Matrix& in)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        const auto & data = in.GetData();

        for (Index i = 0; i < data.size(); ++i)
        {
            norm += data[i] * data[i];
        }

        res = std::sqrt(norm);
        return res;
    }
    Scalar NormP2(const Ellpack& in)
    {
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };

        const auto& count = in.GetCount();
        const auto& data = in.GetData();
        const auto& width = in.GetWidth();

        for (Index i = 0; i < in.GetRows(); ++i)
        {
            for (Index j = 0; j < count[i]; ++j)
            {
                norm += data[i * width + j] * data[i * width + j];
            }
        }

        res = std::sqrt(norm);
        return res;
    }

    Scalar Dot(const Vector& in1, const Vector& in2)
    {
        Scalar res{ 0.0 };

        for (Index i = 0; i < in1.GetRows(); ++i)
        {
            res += in1(i) * in2(i);
        }

        return res;
    }
    Vector Cross(const Vector& in1, const Vector& in2)
    {
        Vector res(3);

        res(0) = in1(1) * in2(2) - in1(2) * in2(1);
        res(1) = in1(2) * in2(0) - in1(0) * in2(2);
        res(2) = in1(0) * in2(1) - in1(1) * in2(0);

        return res;
    }
    
    Vector Merge(const Vector& in1, const Vector& in2)
    {
        NumberRows rows1 = in1.GetRows();
        NumberRows rows2 = in2.GetRows();
        Vector res(rows1 + rows2);

        for (Index i = 0; i < rows1; i++)
        {
            res(i) = in1(i);
        }

        for (Index i = 0; i < rows2; i++)
        {
            res(i + rows1) = in2(i);
        }

        return res;
    }
    void Merge(const Vector& in1, const Vector& in2, Vector& out)
    {
        NumberRows rows1 = in1.GetRows();
        NumberRows rows2 = in2.GetRows();
        
        out.Resize(rows1 + rows2);

        for (Index i = 0; i < rows1; i++)
        {
            out(i) = in1(i);
        }

        for (Index i = 0; i < rows2; i++)
        {
            out(i + rows1) = in2(i);
        }
    }
   
    Scalar Determinant1x1(const Matrix& A)
    {
        return A(0, 0);
    }
    Scalar Determinant2x2(const Matrix& A)
    {
        return A(0,0) * A(1,1) - A(0,1) * A(1,0);
    }
    Scalar Determinant3x3(const Matrix& A)
    {
        return A(0,0) * A(1,1) * A(2,2) + A(0,1) * A(1,2) * A(2,0) + A(0,2) * A(1,0) * A(2,1)
			- A(0, 2) * A(1, 1) * A(2, 0) - A(0, 1) * A(1, 0) * A(2, 2) - A(0, 0) * A(1, 2) * A(2, 1);
    }
    Matrix Inverse1x1(const Matrix& A)
    {
        Matrix res(1, 1);

        res(0, 0) = 1.0 / A(1, 1);

        return res;
    }
    Matrix Inverse2x2(const Matrix& A)
    {
		Matrix res(2, 2);

        res(0, 0) =  A(1, 1);
        res(0, 1) = -A(0, 1);
        res(1, 0) = -A(1, 0);
        res(1, 1) =  A(0, 0);

		return res * (1.0 / Determinant2x2(A));
    }
    Matrix Inverse3x3(const Matrix& A)
    {
        Matrix res(3, 3);

		res(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
		res(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
		res(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);

		res(1, 0) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
		res(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
		res(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);

		res(2, 0) = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
		res(2, 1) = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
		res(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

        return res * (1.0 / Determinant3x3(A));
    }
    Matrix ScaleByVector(const Matrix& A, const Vector& alpha)
    {
        Matrix res(A);

        for (Index i = 0; i < res.GetRows(); ++i)
        {
            for (Index j = 0; j < res.GetCols(); ++j)
            {
                res(i, j) *= alpha(i);
            }
        }

        return res;
    }

    /*
    void DiagonalLinearSystem(const Matrix& A, Vector& x, const Vector& b)
    {
        Index numberRows = A.GetRows();

        for (Index i = 0; i < numberRows; ++i)
        {
            x(i) = b(i) / A(i, i);
        }
    }
    void DiagonalLinearSystem(const Ellpack& A, Vector& x, const Vector& b)
    {
        Index numberRows = A.GetRows();

        for (Index i = 0; i < numberRows; ++i)
        {
            x(i) = b(i) / A(i, i);
        }
    }
    void ForwardLinearSystem(const Matrix& A, Vector& x, const Vector& b)
    {
        Index numberRows = A.GetRows();
        Index numberCols = A.GetCols();
        Scalar dot;

        for (Index i = 0; i < numberRows; ++i)
        {
            dot = 0.0;

            for (Index j = 0; j < numberCols; ++j)
            {
                dot += A(i, j) * x(j);
            }

            x(i) = (b(i) - dot) / A(i, i);
        }
    }
    void ForwardLinearSystem(const Ellpack& A, Vector& x, const Vector& b)
    {
        Index numberRows = A.GetRows();
        Index numberCols = A.GetCols();
        Scalar dot;

        for (Index i = 0; i < numberRows; ++i)
        {
            dot = 0.0;

            for (Index j = 0; j < numberCols; ++j)
            {
                dot += A(i, j) * x(j);
            }

            x(i) = (b(i) - dot) / A(i, i);
        }
    }
    void DecomposeLUP(Matrix& LU, const Matrix& A, Indices& permutation)
    {
        Index numberRows{ A.GetRows() };
        Scalar maxA;
        Scalar absA;

        Index imax{ 0 };
        Index temp{ 0 };
        
        LU = A;
		permutation.resize(numberRows + 1);

        for (Index i = 0; i <= numberRows; i++)
        {
            permutation[i] = i;
        }

        for (Index i = 0; i < numberRows; i++)
        {
            maxA = 0.0;
            imax = 0;

            for (Index k = i; k < numberRows; k++)
            {
                absA = std::abs(A(k, i));
                if (absA > maxA)
                {
                    maxA = absA;
                    imax = k;
                }
            }

            if (imax != i)
            {
                temp = permutation[i];
                permutation[i] = permutation[imax];
                permutation[imax] = temp;

                LU.SwapRows(i, imax);

                permutation[numberRows]++;
            }

            for (Index j = i + 1; j < numberRows; j++)
            {
                LU(j, i) /= LU(i, i);

                for (Index k = i + 1; k < numberRows; k++)
                {
                    LU(j, k) -= LU(j, i) * LU(i, k);
                }
            }
        }
    }
    void InverseLUP(Matrix& IA, const Matrix& LU, const Indices& permutation)
    {
        Index numberRows{ LU.GetRows() };

        for (Index j = 0; j < numberRows; j++) 
        {
            for (Index i = 0; i < numberRows; i++)
            {
                IA(i, j) = permutation[i] == j ? 1.0 : 0.0;

                for (Index k = 0; k < i; k++)
                {
                    IA(i, j) -= LU(i, k) * IA(k, j);
                }
            }

            for (Index i = numberRows - 1 + 1; i > 0; i--) 
            {
                for (Index k = i; k < numberRows; k++)
                {
                    IA(i - 1, j) -= LU(i - 1, k) * IA(k, j);
                }

                IA(i - 1, j) /= LU(i - 1, i - 1);
            }
        }
    }
    void DirectLUP(const Matrix& LU, Vector& x, const Vector& b, const Indices& permutation)
    {
        Index numberRows = LU.GetRows();

        for (Index i = 0; i < numberRows; i++)
        {
            x(i) = b(permutation[i]);

            for (Index k = 0; k < i; k++)
            {
                x(i) -= LU(i, k) * x(k);
            }
        }

        for (Index i = numberRows; i > 0; i--)
        {
            for (Index k = i; k < numberRows; k++)
            {
                x(i - 1) -= LU(i - 1, k) * x(k);
            }

            x(i - 1) /= LU(i - 1, i - 1);
        }
    }
    void Direct(const Matrix& A, Vector& x, const Vector& b)
    {
        Index numberRows{ A.GetRows() };
        Index numberCols{ A.GetCols() };

        Matrix LU(numberRows, numberCols);
        Indices permutation(numberRows + 1);

        x.Resize(numberRows);

        DecomposeLUP(LU, A, permutation);
        DirectLUP(LU, x, b, permutation);
    }
    */
    void WriteToFile(const Vector& vec, const String& fileName)
    {
        File file;

        file.SetName(fileName);
        file.SetMode(utils::file::Write);

        if (file.Open() != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be created");
        }
        
        auto output = ListVector(vec);

        file.Write(output);
    }
    /*void WriteToFile(const Matrix& mat, const String& fileName)
    {
        File file;

        file.SetName(fileName);
        file.SetMode(utils::file::Write);

        if (file.Open() != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be created");
        }

        auto output = ListMatrix(mat);

        file.Write(output);
    }
    void WriteToFile(const Ellpack& mat, const String& fileName)
    {
        File file;

        file.SetName(fileName);
        file.SetMode(utils::file::Write);

        if (file.Open() != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be created");
        }

        auto output = ListMatrix(mat);

        file.Write(output);
    }*/

    Status ReadFromFile(Vector& output, const String& fileName)
    {
        File file;
        String line;
        Status status;
        Strings table;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);
        
        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        auto stream = static_cast<std::istringstream>(file.GetFull());
        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                table.push_back(line);
            }
        }

        output.Resize(table.size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            output(i) = utils::string::ConvertTo<Scalar>(table[i]);
        }

        return EILIG_SUCCESS;
    }
    Status ReadFromFile(Matrix& output, const String& fileName)
    {
        File file;
        String line;
        Status status;
        Strings split;
        std::vector<Strings> table;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }
        
        auto stream = static_cast<std::istringstream>(file.GetFull());
        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                split = utils::string::Split(line, {' ', ';', '\t'});
                table.push_back(split);
            }
        }

        output.Resize(table.size(), table[0].size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            for (Index j = 0; j < output.GetCols(); j++)
            {
                output.Equal(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
            }
        }

        return EILIG_SUCCESS;
    }
    Status ReadFromFile(Ellpack& output, const String& fileName)
    {
        File file;
        String line;
        Status status;
        std::vector<Strings> table;
        Strings split;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        auto stream = static_cast<std::istringstream>(file.GetFull());
        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                split = utils::string::Split(line, { ' ', ';', '\t' });
                table.push_back(split);
            }
        }

        output.Resize(table.size(), table[0].size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            for (Index j = 0; j < output.GetCols(); j++)
            {
                output.Equal(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
            }
        }

        return EILIG_SUCCESS;
    }

#ifdef EILIG_ENABLE_OPENCL
    Scalar NormMax(const opencl::Vector& in)
    {
        club::Error error;
        Index numberRows = in.GetRows();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalars partial;

        const auto& dimension = in.GetKernel()->kVectorNormMax_->GetDim();
        const auto& localSize = opencl::GetContext()->GetLocalSize(in.GetDeviceIndex(), dimension);

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);

        opencl::BufferPtr partialGPU = club::CreateBuffer(opencl::GetContext(), sizeof(Scalar) * ngroups);

        in.GetKernel()->kVectorNormMax_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernel()->kVectorNormMax_->SetArg(1, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernel()->kVectorNormMax_->SetArg(2, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernel()->kVectorNormMax_->SetArg(3, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(opencl::GetContext()->GetQueues()[in.GetDeviceIndex()],
            in.GetKernel()->kVectorNormMax_->GetKernel(),
            in.GetKernel()->kVectorNormMax_->GetDim(), NULL, globalSize,
            &localSize[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
        }

        partial.resize(ngroups);
        partialGPU->Read(opencl::GetContext()->GetQueues()[in.GetDeviceIndex()], 0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        res = *std::max_element(partial.begin(), partial.end());

        return res;
    }
    Scalar NormP(const opencl::Vector& in, Scalar p)
    {
        club::Error error;
        Index numberRows = in.GetRows();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };
        Scalars partial;

        const auto& dimension = in.GetKernel()->kVectorNormP_->GetDim();
        const auto& localSize = opencl::GetContext()->GetLocalSize(in.GetDeviceIndex(), dimension);

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);

        opencl::BufferPtr partialGPU = club::CreateBuffer(opencl::GetContext(), sizeof(Scalar) * ngroups);

        in.GetKernel()->kVectorNormP_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernel()->kVectorNormP_->SetArg(1, sizeof(Scalar), &p);
        in.GetKernel()->kVectorNormP_->SetArg(2, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernel()->kVectorNormP_->SetArg(3, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernel()->kVectorNormP_->SetArg(4, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(opencl::GetContext()->GetQueues()[in.GetDeviceIndex()],
            in.GetKernel()->kVectorNormP_->GetKernel(),
            in.GetKernel()->kVectorNormP_->GetDim(), NULL, globalSize,
            &localSize[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
        }

        partial.resize(ngroups);
        partialGPU->Read(opencl::GetContext()->GetQueues()[in.GetDeviceIndex()], 0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::pow(norm, 1. / p);
        return res;
    }
    /*Scalar NormP(const opencl::Ellpack& in, Scalar p)
    {
        club::Error error;
        Index numberRows = in.GetRows();
        Index numberCols = in.GetCols();
        Index width = in.GetWidth();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };
        Scalars partial;

        const auto& localSize = in.GetKernel()->kEllpackNormP_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernel()->context_, sizeof(Scalar) * ngroups);

        in.GetKernel()->kEllpackNormP_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernel()->kEllpackNormP_->SetArg(1, sizeof(Index), &numberCols);
        in.GetKernel()->kEllpackNormP_->SetArg(2, sizeof(Index), &width);
        in.GetKernel()->kEllpackNormP_->SetArg(3, sizeof(Scalar), &p);
        in.GetKernel()->kEllpackNormP_->SetArg(4, sizeof(cl_mem), &in.GetCountGPU()->Get());
        in.GetKernel()->kEllpackNormP_->SetArg(5, sizeof(cl_mem), &in.GetPositionGPU()->Get());
        in.GetKernel()->kEllpackNormP_->SetArg(6, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernel()->kEllpackNormP_->SetArg(7, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernel()->kEllpackNormP_->SetArg(8, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernel()->context_->GetQueue(),
            in.GetKernel()->kEllpackNormP_->GetKernel(),
            in.GetKernel()->kEllpackNormP_->GetDim(), NULL, globalSize,
            &in.GetKernel()->kEllpackNormP_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::pow(norm, 1. / p);
        return res;
    }*/
    Scalar NormP2(const opencl::Vector& in)
    {
        club::Error error;
        Index numberRows = in.GetRows();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };
        Scalars partial;

        const auto& dimension = in.GetKernel()->kVectorNormP2_->GetDim();
        const auto& localSize = opencl::GetContext()->GetLocalSize(in.GetDeviceIndex(), dimension);

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);;

        opencl::BufferPtr partialGPU = club::CreateBuffer(opencl::GetContext(), sizeof(Scalar) * ngroups);

        in.GetKernel()->kVectorNormP2_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernel()->kVectorNormP2_->SetArg(1, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernel()->kVectorNormP2_->SetArg(2, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernel()->kVectorNormP2_->SetArg(3, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(opencl::GetContext()->GetQueues()[in.GetDeviceIndex()],
            in.GetKernel()->kVectorNormP2_->GetKernel(),
            in.GetKernel()->kVectorNormP2_->GetDim(), NULL, globalSize,
            &localSize[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
        }

        partial.resize(ngroups);
        partialGPU->Read(opencl::GetContext()->GetQueues()[in.GetDeviceIndex()], 0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::sqrt(norm);
        return res;
    }
    /*Scalar NormP2(const opencl::Ellpack& in)
    {
        club::Error error;
        Index numberRows = in.GetRows();
        Index numberCols = in.GetCols();
        Index width = in.GetWidth();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };
        Scalars partial;

        const auto& localSize = in.GetKernel()->kEllpackNormP2_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);;

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernel()->context_, sizeof(Scalar) * ngroups);

        in.GetKernel()->kEllpackNormP2_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernel()->kEllpackNormP2_->SetArg(1, sizeof(Index), &numberCols);
        in.GetKernel()->kEllpackNormP2_->SetArg(2, sizeof(Index), &width);
        in.GetKernel()->kEllpackNormP2_->SetArg(3, sizeof(cl_mem), &in.GetCountGPU()->Get());
        in.GetKernel()->kEllpackNormP2_->SetArg(4, sizeof(cl_mem), &in.GetPositionGPU()->Get());
        in.GetKernel()->kEllpackNormP2_->SetArg(5, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernel()->kEllpackNormP2_->SetArg(6, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernel()->kEllpackNormP2_->SetArg(7, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernel()->context_->GetQueue(),
            in.GetKernel()->kEllpackNormP2_->GetKernel(),
            in.GetKernel()->kEllpackNormP2_->GetDim(), NULL, globalSize,
            &in.GetKernel()->kEllpackNormP2_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::sqrt(norm);
        return res;
    }*/

    Scalar Dot(const opencl::Vector& in1, const opencl::Vector& in2)
    {
        club::Error error;
        Index numberRows = in1.GetRows();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalars partial;

        const auto& dimension = in1.GetKernel()->kVectorDot_->GetDim();
        const auto& localSize = opencl::GetContext()->GetLocalSize(in1.GetDeviceIndex(), dimension);

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);;

        opencl::BufferPtr partialGPU = club::CreateBuffer(opencl::GetContext(), sizeof(Scalar) * ngroups);

        in1.GetKernel()->kVectorDot_->SetArg(0, sizeof(Index), &numberRows);
        in1.GetKernel()->kVectorDot_->SetArg(1, sizeof(cl_mem), &in1.GetDataGPU()->Get());
        in1.GetKernel()->kVectorDot_->SetArg(2, sizeof(cl_mem), &in2.GetDataGPU()->Get());
        in1.GetKernel()->kVectorDot_->SetArg(3, sizeof(cl_mem), &partialGPU->Get());
        in1.GetKernel()->kVectorDot_->SetArg(4, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(opencl::GetContext()->GetQueues()[in1.GetDeviceIndex()],
            in1.GetKernel()->kVectorDot_->GetKernel(),
            in1.GetKernel()->kVectorDot_->GetDim(), NULL, globalSize,
            &localSize[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, utils::string::Format("Enqueueing kernel: {}", club::messages.at(error)));
        }

        partial.resize(ngroups);
        partialGPU->Read(opencl::GetContext()->GetQueues()[in1.GetDeviceIndex()], 0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            res += partial[i];
        }

        return res;
    }

    void WriteToFile(const opencl::Vector& vec, const String& fileName)
    {
        File file;

        file.SetName(fileName);
        file.SetMode(utils::file::Write);

        if (file.Open() != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be created");
        }

        auto output = ListVector(vec);

        file.Write(output);
    }
    /*void WriteToFile(const opencl::Ellpack& mat, const String& fileName)
    {
        File file;

        file.SetName(fileName);
        file.SetMode(utils::file::Write);

        if (file.Open() != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be created");
        }

        auto output = ListMatrix(mat);

        file.Write(output);
    }*/

    Status ReadFromFile(opencl::Vector& output, const String& fileName)
    {
        File file;
        String line;
        Status status;
        Strings table;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        auto stream = static_cast<std::istringstream>(file.GetFull());
        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                table.push_back(line);
            }
        }

        output.Resize(table.size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            output(i) = utils::string::ConvertTo<Scalar>(table[i]);
        }

        return EILIG_SUCCESS;
    }
    /*Status ReadFromFile(opencl::Ellpack& output, const String& fileName)
    {
        File file;
        String line;
        Status status;
        std::vector<Strings> table;
        Strings split;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        auto stream = static_cast<std::istringstream>(file.GetFull());
        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                split = utils::string::Split(line, { ' ', ';', '\t' });
                table.push_back(split);
            }
        }

        output.Resize(table.size(), table[0].size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            for (Index j = 0; j < output.GetCols(); j++)
            {
                output.Equal(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
            }
        }

        return EILIG_SUCCESS;
    }*/
#endif

} /* namespace eilig */