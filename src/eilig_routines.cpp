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
        Scalar res{ std::abs(in(0)) };
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
    Scalar DeterminantLUP(const Matrix& LU, const Indices& permutation)
    {
        Index numberRows{ LU.GetRows() };
        Scalar det{ LU(0, 0) };

        for (Index i = 1; i < numberRows; i++)
        {
            det *= LU(i, i);
        }

        return (permutation[numberRows] - numberRows) % 2 == 0 ? det : -det;
    }
    Scalar Determinant(const Matrix& A)
    {
        Index numberRows{ A.GetRows() };
        Index numberCols{ A.GetCols() };

        Matrix LU(numberRows, numberCols);
        Indices permutation(numberRows + 1 );

        DecomposeLUP(LU, A, permutation);

        return DeterminantLUP(LU, permutation);
    }
    Matrix Inverse(const Matrix& A)
    {
        Index numberRows{ A.GetRows() };
        Index numberCols{ A.GetCols() };

        Matrix LU(numberRows, numberCols);
        Matrix IA(numberRows, numberCols);
        Indices permutation(numberRows + 1);

        DecomposeLUP(LU, A, permutation);
        InverseLUP(IA, LU, permutation);

        return IA;
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
    Vector Solve(const Matrix& A, const Vector& b)
    {
        Index numberRows{ A.GetRows() };
        Index numberCols{ A.GetCols() };

        Vector x(numberRows);
        Matrix LU(numberRows, numberCols);
        Indices permutation(numberRows + 1);

        DecomposeLUP(LU, A, permutation);
        DirectLUP(LU, x, b, permutation);
    
        return x;
    }
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
		permutation.resize(numberRows);

        for (Index i = 0; i < numberRows; i++)
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
    Status IterativeCG(const Ellpack& A, Vector& x, const Vector& b, CallbackIterative callbackIterative)
    {
        Scalar alpha{ 0.0 };
        Scalar beta{ 0.0 };
        Scalar rho0{ 0.0 };

        Index numberRows = A.GetRows();
        Index iteration = { 0 };

        Vector x0(numberRows);
        Vector r0(numberRows);
        Vector p0(numberRows);
        Vector x1(numberRows);
        Vector r1(numberRows);

        if (callbackIterative == nullptr)
        {
            logger::Error(headerEilig, "Invalid callback (null pointer)");
            return EILIG_NULLPTR;
        }

        x0 = x;
        r0 = b - A * x0;

        if (utils::math::IsAlmostEqual(NormP2(r0), 0.0, 5))
        {
            x0 = 1.0;
            r0 = b - A * x0;
        }

        p0 = r0;

        for (;;)
        {
            iteration++;

            alpha = Dot(r0, r0) / Dot(p0, A * p0);

            x1 = x0 + alpha * p0 ;
            r1 = r0 - alpha * (A * p0);

            auto residual = NormP2(r1);
            auto status = callbackIterative(iteration, residual);

            switch (status)
            {
            case EILIG_SUCCESS:
                x = x1;
                return status;
            case EILIG_NOT_CONVERGED:
                x = x1;
                return status;
            case EILIG_STOP:
                x = x1;
                return status;
            case EILIG_CONTINUE:
                break;
            }

            beta = Dot(r1, r1) / Dot(r0, r0);
            p0 = r1 + beta * p0;

            x0 = x1;
            r0 = r1;
        }
        
        return EILIG_NOT_CONVERGED;
    }
    Status IterativeBiCGStab(const Ellpack& A, Vector& x, const Vector& b, CallbackIterative callbackIterative)
    {
        Scalar alpha{ 0.0 };
        Scalar beta{ 0.0 };
        Scalar omega{ 0.0 };

        Index numberRows = A.GetRows();
        Index iteration { 0 };

        Vector x0(numberRows);
        Vector r0(numberRows);
        Vector p0(numberRows);
        Vector s0(numberRows);
        Vector h0(numberRows);
        Vector t0(numberRows);
        Vector v0(numberRows);
        Vector x1(numberRows);
        Vector p1(numberRows);
        Vector r1(numberRows);
        Vector r2(numberRows);

        if (callbackIterative == nullptr)
        {
            logger::Error(headerEilig, "Invalid callback (null pointer)");
            return EILIG_NULLPTR;
        }

        x0 = x;       
        r0 = b - A * x0;

        if (utils::math::IsAlmostEqual(NormP2(r0), 0.0, 5))
        {
            x0 = x + 1.0;
            r0 = b - A * x0;
        }

        p0 = r0;
        r1 = r0;
        
        for (;;)
        {
            iteration++;

            v0 = A * p0;
            alpha = Dot(r1, r0) / Dot(v0, r0);
            h0 = x0 + alpha * p0;
            s0 = r1 - alpha * v0;

            auto residual = NormP2(s0);
            auto status = callbackIterative(iteration, residual);

            switch (status)
            {
            case EILIG_SUCCESS:
                x = h0;
                return status;
            }

            t0 = A * s0;
            omega = Dot(t0, s0) / Dot(t0, t0);
            
            x1 = x0 + alpha * p0 + omega * s0;
            r2 = s0 - omega * t0;

            residual = NormP2(r2);
            status = callbackIterative(iteration, residual);

            switch (status)
            {
            case EILIG_SUCCESS:
                x = x1;
                return status;
            case EILIG_NOT_CONVERGED:
                x = x1;
                return status;
            case EILIG_STOP:
                x = x1;
                return status;
            case EILIG_CONTINUE:
                break;
            }

            beta = (Dot(r2, r0) / Dot(r1, r0)) * (alpha / omega);            
            p1 = r2 + beta * (p0 - omega * v0);
            r1 = r2;

            p0 = p1;
            x0 = x1;
        }

        return EILIG_NOT_CONVERGED;
    }

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
    void WriteToFile(const Matrix& mat, const String& fileName)
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
    }
    Status ReadFromFile(Vector& output, const String& fileName)
    {
        File file;
        String input;
        String line;
        Status status;
        std::vector<String> table;
        std::istringstream stream;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);
        
        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        input = file.GetFull();
        stream = static_cast<std::istringstream>(input);

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
        String input;
        String line;
        Status status;
        std::vector<std::vector<String>> table;
        std::vector<String> split;
        std::istringstream stream;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        input = file.GetFull();
        stream = static_cast<std::istringstream>(input);

        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                split = utils::string::Split(line);
                table.push_back(split);
            }
        }

        output.Resize(table.size(), table[0].size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            for (Index j = 0; j < output.GetCols(); j++)
            {
                output.SetValue(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
            }
        }

        return EILIG_SUCCESS;
    }
    Status ReadFromFile(Ellpack& output, const String& fileName)
    {
        File file;
        String input;
        String line;
        Status status;
        std::vector<std::vector<String>> table;
        std::vector<String> split;
        std::istringstream stream;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        input = file.GetFull();
        stream = static_cast<std::istringstream>(input);

        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                split = utils::string::Split(line);
                table.push_back(split);
            }
        }

        output.Resize(table.size(), table[0].size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            for (Index j = 0; j < output.GetCols(); j++)
            {
                output.SetValue(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
            }
        }

        return EILIG_SUCCESS;
    }
    String ListVector(const Vector& vector)
    {
        String output{};
        Index numberRows = vector.GetRows();

        //logger::Info(headerEilig, "Vector (%zu x 1):", vector.GetRows());

        for (Index i = 0; i < numberRows; ++i)
        {
            output += utils::string::Format("%14.5e\n", vector(i));
        }

        return output;
    }
    String ListMatrix(const Matrix& matrix)
    {
        String output{};
        Index numberRows = matrix.GetRows();
        Index numberCols = matrix.GetCols();

        //logger::Info(headerEilig, "Matrix (%zu x %zu):", matrix.GetRows(), matrix.GetCols());

        for (Index i = 0; i < numberRows; ++i)
        {
            for (Index j = 0; j < numberCols; ++j)
            {
                output += utils::string::Format("%14.5e", matrix(i, j));
            }
            
            output += "\n";
        }

        return output;
    }
    String ListMatrix(const Ellpack& matrix)
    {
        String output{};
        Index numberRows = matrix.GetRows();
        Index numberCols = matrix.GetCols();
        Index width = matrix.GetWidth();

        const auto& count = matrix.GetCount();
        const auto& data = matrix.GetData();
        const auto& position = matrix.GetPosition();

        //logger::Info(headerEilig, "Matrix Ellpack (%zu x %zu):", matrix.GetRows(), matrix.GetCols());

        for (Index i = 0; i < numberRows; ++i)
        {
            Index k = 0;

            for (Index j = 0; j < numberCols; j++)
            {
                if ((k < count[i]) && (position[i * width + k] == j))
                {
                    output += utils::string::Format("%14.5e", data[i * width + k]);
                    k++;
                    continue;
                }

                output += utils::string::Format("%14.5e", 0.0);
            }

            output += "\n";
        }

        return output;
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

        const auto& localSize = in.GetKernels()->kVectorNormMax_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernels()->context_, sizeof(Scalar) * ngroups);

        in.GetKernels()->kVectorNormMax_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernels()->kVectorNormMax_->SetArg(1, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernels()->kVectorNormMax_->SetArg(2, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernels()->kVectorNormMax_->SetArg(3, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernels()->context_->GetQueue(),
            in.GetKernels()->kVectorNormMax_->GetKernel(),
            in.GetKernels()->kVectorNormMax_->GetDim(), NULL, globalSize,
            &in.GetKernels()->kVectorNormMax_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

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

        const auto& localSize = in.GetKernels()->kVectorNormP_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernels()->context_, sizeof(Scalar) * ngroups);

        in.GetKernels()->kVectorNormP_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernels()->kVectorNormP_->SetArg(1, sizeof(Scalar), &p);
        in.GetKernels()->kVectorNormP_->SetArg(2, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernels()->kVectorNormP_->SetArg(3, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernels()->kVectorNormP_->SetArg(4, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernels()->context_->GetQueue(),
            in.GetKernels()->kVectorNormP_->GetKernel(),
            in.GetKernels()->kVectorNormP_->GetDim(), NULL, globalSize,
            &in.GetKernels()->kVectorNormP_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::pow(norm, 1. / p);
        return res;
    }
    Scalar NormP(const opencl::Ellpack& in, Scalar p)
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

        const auto& localSize = in.GetKernels()->kEllpackNormP_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernels()->context_, sizeof(Scalar) * ngroups);

        in.GetKernels()->kEllpackNormP_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernels()->kEllpackNormP_->SetArg(1, sizeof(Index), &numberCols);
        in.GetKernels()->kEllpackNormP_->SetArg(2, sizeof(Index), &width);
        in.GetKernels()->kEllpackNormP_->SetArg(3, sizeof(Scalar), &p);
        in.GetKernels()->kEllpackNormP_->SetArg(4, sizeof(cl_mem), &in.GetCountGPU()->Get());
        in.GetKernels()->kEllpackNormP_->SetArg(5, sizeof(cl_mem), &in.GetPositionGPU()->Get());
        in.GetKernels()->kEllpackNormP_->SetArg(6, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernels()->kEllpackNormP_->SetArg(7, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernels()->kEllpackNormP_->SetArg(8, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernels()->context_->GetQueue(),
            in.GetKernels()->kEllpackNormP_->GetKernel(),
            in.GetKernels()->kEllpackNormP_->GetDim(), NULL, globalSize,
            &in.GetKernels()->kEllpackNormP_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::pow(norm, 1. / p);
        return res;
    }
    Scalar NormP2(const opencl::Vector& in)
    {
        club::Error error;
        Index numberRows = in.GetRows();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalar norm{ 0.0 };
        Scalars partial;

        const auto& localSize = in.GetKernels()->kVectorNormP2_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);;

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernels()->context_, sizeof(Scalar) * ngroups);

        in.GetKernels()->kVectorNormP2_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernels()->kVectorNormP2_->SetArg(1, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernels()->kVectorNormP2_->SetArg(2, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernels()->kVectorNormP2_->SetArg(3, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernels()->context_->GetQueue(),
            in.GetKernels()->kVectorNormP2_->GetKernel(),
            in.GetKernels()->kVectorNormP2_->GetDim(), NULL, globalSize,
            &in.GetKernels()->kVectorNormP2_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::sqrt(norm);
        return res;
    }
    Scalar NormP2(const opencl::Ellpack& in)
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

        const auto& localSize = in.GetKernels()->kEllpackNormP2_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);;

        opencl::BufferPtr partialGPU = club::CreateBuffer(in.GetKernels()->context_, sizeof(Scalar) * ngroups);

        in.GetKernels()->kEllpackNormP2_->SetArg(0, sizeof(Index), &numberRows);
        in.GetKernels()->kEllpackNormP2_->SetArg(1, sizeof(Index), &numberCols);
        in.GetKernels()->kEllpackNormP2_->SetArg(2, sizeof(Index), &width);
        in.GetKernels()->kEllpackNormP2_->SetArg(3, sizeof(cl_mem), &in.GetCountGPU()->Get());
        in.GetKernels()->kEllpackNormP2_->SetArg(4, sizeof(cl_mem), &in.GetPositionGPU()->Get());
        in.GetKernels()->kEllpackNormP2_->SetArg(5, sizeof(cl_mem), &in.GetDataGPU()->Get());
        in.GetKernels()->kEllpackNormP2_->SetArg(6, sizeof(cl_mem), &partialGPU->Get());
        in.GetKernels()->kEllpackNormP2_->SetArg(7, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in.GetKernels()->context_->GetQueue(),
            in.GetKernels()->kEllpackNormP2_->GetKernel(),
            in.GetKernels()->kEllpackNormP2_->GetDim(), NULL, globalSize,
            &in.GetKernels()->kEllpackNormP2_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            norm += partial[i];
        }

        res = std::sqrt(norm);
        return res;
    }
    Scalar Dot(const opencl::Vector& in1, const opencl::Vector& in2)
    {
        club::Error error;
        Index numberRows = in1.GetRows();
        Index globalSize[1];
        Index ngroups{ 0 };
        Scalar res{ 0.0 };
        Scalars partial;

        const auto& localSize = in1.GetKernels()->kVectorDot_->GetLocalSize();

        globalSize[0] = localSize[0] * (numberRows / localSize[0] + (numberRows % localSize[0] != 0 ? 1 : 0));
        ngroups = (numberRows % localSize[0]) > 0 ? (numberRows / localSize[0] + 1) : (numberRows / localSize[0]);;

        opencl::BufferPtr partialGPU = club::CreateBuffer(in1.GetKernels()->context_, sizeof(Scalar) * ngroups);

        in1.GetKernels()->kVectorDot_->SetArg(0, sizeof(Index), &numberRows);
        in1.GetKernels()->kVectorDot_->SetArg(1, sizeof(cl_mem), &in1.GetDataGPU()->Get());
        in1.GetKernels()->kVectorDot_->SetArg(2, sizeof(cl_mem), &in2.GetDataGPU()->Get());
        in1.GetKernels()->kVectorDot_->SetArg(3, sizeof(cl_mem), &partialGPU->Get());
        in1.GetKernels()->kVectorDot_->SetArg(4, localSize[0] * sizeof(Scalar), NULL);

        error = clEnqueueNDRangeKernel(in1.GetKernels()->context_->GetQueue(),
            in1.GetKernels()->kVectorDot_->GetKernel(),
            in1.GetKernels()->kVectorDot_->GetDim(), NULL, globalSize,
            &in1.GetKernels()->kVectorDot_->GetLocalSize()[0], 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            logger::Error(headerEilig, "Enqueueing kernel: " + club::messages.at(error));
        }

        partial.resize(ngroups);
        partialGPU->Read(0, sizeof(Scalar) * ngroups, &partial[0], CL_TRUE);

        for (Index i = 0; i < partial.size(); i++)
        {
            res += partial[i];
        }

        return res;
    }
    Status IterativeCG(const opencl::Ellpack& A, opencl::Vector& x, const opencl::Vector& b, CallbackIterative callbackIterative)
    {
        Scalar alpha{ 0.0 };
        Scalar beta{ 0.0 };
        Scalar rho0{ 0.0 };

        Index numberRows = A.GetRows();
        Index iteration = { 0 };

        opencl::Vector x0(b.GetKernels(), numberRows);
        opencl::Vector r0(b.GetKernels(), numberRows);
        opencl::Vector p0(b.GetKernels(), numberRows);
        opencl::Vector x1(b.GetKernels(), numberRows);
        opencl::Vector r1(b.GetKernels(), numberRows);
    
        if (callbackIterative == nullptr)
        {
            logger::Error(headerEilig, "Invalid callback (null pointer)");
            return EILIG_NULLPTR;
        }

        x0 = x;
        r0 = b - A * x0;

        if (utils::math::IsAlmostEqual(NormP2(r0), 0.0, 5))
        {
            x0 = x + 1.0;
            r0 = b - A * x0;
        }

        p0 = r0;

        for (;;)
        {
            iteration++;

            alpha = Dot(r0, r0) / Dot(p0, A * p0);

            x1 = x0 + alpha * p0;
            r1 = r0 - alpha * (A * p0);

            auto residual = NormP2(r1);
            auto status = callbackIterative(iteration, residual);

            switch (status)
            {
            case EILIG_SUCCESS:
                x = x1;
                return status;
            case EILIG_NOT_CONVERGED:
                x = x1;
                return status;
            case EILIG_STOP:
                x = x1;
                return status;
            case EILIG_CONTINUE:
                break;
            }

            beta = Dot(r1, r1) / Dot(r0, r0);
            p0 = r1 + beta * p0;

            x0 = x1;
            r0 = r1;
        }

        return EILIG_NOT_CONVERGED;
    }
    Status IterativeBiCGStab(const opencl::Ellpack& A, opencl::Vector& x, const opencl::Vector& b, CallbackIterative callbackIterative)
    {
        Scalar alpha{ 0.0 };
        Scalar beta{ 0.0 };
        Scalar omega{ 0.0 };

        Index numberRows = A.GetRows();
        Index iteration{ 0 };

        opencl::Vector x0(b.GetKernels(), numberRows);
        opencl::Vector r0(b.GetKernels(), numberRows);
        opencl::Vector p0(b.GetKernels(), numberRows);
        opencl::Vector s0(b.GetKernels(), numberRows);
        opencl::Vector h0(b.GetKernels(), numberRows);
        opencl::Vector t0(b.GetKernels(), numberRows);
        opencl::Vector v0(b.GetKernels(), numberRows);
        opencl::Vector x1(b.GetKernels(), numberRows);
        opencl::Vector p1(b.GetKernels(), numberRows);
        opencl::Vector r1(b.GetKernels(), numberRows);
        opencl::Vector r2(b.GetKernels(), numberRows);

        if (callbackIterative == nullptr)
        {
            logger::Error(headerEilig, "Invalid callback (null pointer)");
            return EILIG_NULLPTR;
        }

        x0 = x;
        r0 = b - A * x0;

        if (utils::math::IsAlmostEqual(NormP2(r0), 0.0, 5))
        {
            x0 = x + 1.0;
            r0 = b - A * x0;
        }

        p0 = r0;
        r1 = r0;

        for (;;)
        {
            iteration++;

            v0 = A * p0;
            alpha = Dot(r1, r0) / Dot(v0, r0);
            h0 = x0 + alpha * p0;
            s0 = r1 - alpha * v0;

            auto residual = NormP2(s0);
            auto status = callbackIterative(iteration, residual);

            switch (status)
            {
            case EILIG_SUCCESS:
                x = h0;
                return status;
            }

            t0 = A * s0;
            omega = Dot(t0, s0) / Dot(t0, t0);

            x1 = x0 + alpha * p0 + omega * s0;
            r2 = s0 - omega * t0;

            residual = NormP2(r2);
            status = callbackIterative(iteration, residual);

            switch (status)
            {
            case EILIG_SUCCESS:
                x = x1;
                return status;
            case EILIG_NOT_CONVERGED:
                x = x1;
                return status;
            case EILIG_STOP:
                x = x1;
                return status;
            case EILIG_CONTINUE:
                break;
            }

            beta = (Dot(r2, r0) / Dot(r1, r0)) * (alpha / omega);
            p1 = r2 + beta * (p0 - omega * v0);
            r1 = r2;

            p0 = p1;
            x0 = x1;
        }

        return EILIG_NOT_CONVERGED;
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
    void WriteToFile(const opencl::Ellpack& mat, const String& fileName)
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
    Status ReadFromFile(opencl::Vector& output, const String& fileName)
    {
        File file;
        String input;
        String line;
        Status status;
        std::vector<String> table;
        std::istringstream stream;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        input = file.GetFull();
        stream = static_cast<std::istringstream>(input);

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
    Status ReadFromFile(opencl::Ellpack& output, const String& fileName)
    {
        File file;
        String input;
        String line;
        Status status;
        std::vector<std::vector<String>> table;
        std::vector<String> split;
        std::istringstream stream;

        file.SetName(fileName);
        file.SetMode(utils::file::Read);

        status = file.Open();
        if (status != utils::file::UTILS_SUCCESS)
        {
            logger::Error(headerEilig, "File could not be opened");
            return EILIG_INVALID_FILE;
        }

        input = file.GetFull();
        stream = static_cast<std::istringstream>(input);

        while (std::getline(stream, line))
        {
            if (!utils::string::IsEmpty(line))
            {
                split = utils::string::Split(line);
                table.push_back(split);
            }
        }

        output.Resize(table.size(), table[0].size());

        for (Index i = 0; i < output.GetRows(); i++)
        {
            for (Index j = 0; j < output.GetCols(); j++)
            {
                output.SetValue(i, j, utils::string::ConvertTo<Scalar>(table[i][j]));
            }
        }

        return EILIG_SUCCESS;
    }
    String ListVector(const opencl::Vector& vector)
    {
        String output{};
        Scalars data{};
        Index numberRows = vector.GetRows();

        const auto& dataGPU = vector.GetDataGPU();

        data.resize(numberRows);
        dataGPU->Read(0, sizeof(Scalar) * numberRows, &data[0], CL_TRUE);

        //logger::Info(headerEilig, "Vector CL (%zu x 1):", vector.GetRows());

        for (Index i = 0; i < numberRows; ++i)
        {
            output += utils::string::Format("%14.5e\n", data[i]);
        }

        return output;
    }
    String ListMatrix(const opencl::Ellpack& matrix)
    {
        String output{};
        Scalars data{};
        Indices position{};
        Indices count{};

        Index numberRows = matrix.GetRows();
        Index numberCols = matrix.GetCols();
        Index width = matrix.GetWidth();
        const auto& countGPU = matrix.GetCountGPU();
        const auto& dataGPU = matrix.GetDataGPU();
        const auto& positionGPU = matrix.GetPositionGPU();

        count.resize(numberRows);
        data.resize(numberRows * width);
        position.resize(numberRows * width);

        countGPU->Read(0, sizeof(Index) * numberRows, &count[0], CL_TRUE);
        dataGPU->Read(0, sizeof(Scalar) * numberRows * width, &data[0], CL_TRUE);
        positionGPU->Read(0, sizeof(Index) * numberRows * width, &position[0], CL_TRUE);

        //logger::Info(headerEilig, "Matrix Ellpack CL (%zu x %zu):", matrix.GetRows(), matrix.GetCols());

        for (Index i = 0; i < numberRows; ++i)
        {
            Index k = 0;

            for (Index j = 0; j < numberCols; j++)
            {
                if ( (k < count[i]) && (position[i * width + k] == j))
                {
                    output += utils::string::Format("%14.5e", data[i * width + k]);
                    k++;
                    continue;
                }

                output += utils::string::Format("%14.5e", 0.0);
            }
            output += "\n";
        }

        return output;
    }
#endif    

} /* namespace eilig */

/*
    Status IterativeJacobi(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol, Index itmax)
    {
        Status status = EILIG_RUNNING;
        Index numberRows = A.GetRows();
        Index numberCols = A.GetCols();
        Index max = static_cast<Index>(std::pow(numberRows, 1.5));
        Index iteration = { 0 };

        Scalar residualNorm;

        Vector x1(numberRows);
        Vector x2(numberRows);
        Vector u(numberRows);
        Vector v(numberRows);
        Vector aux(numberRows);
        Vector residual(numberRows);
        Ellpack M;
        Ellpack N;

        if (itmax != 0)
        {
            max = itmax;
        }

        if (rtol <= 0.)
        {
            status = EILIG_INVALID_TOLERANCE;
            logger::Error(headerEilig, "Convergence tolerance must be a positive real number");
            return status;
        }

        x1 = 0.;
        M = A.Diagonal();
        N = -(A.Lower(false) + A.Upper(false));

        for (iteration = 1; iteration < max; ++iteration)
        {
            Mul(aux, N, x1);
            Add(u, aux, b);
            DiagonalLinearSystem(x2, M, u);

            Mul(aux, A, x2);
            Sub(residual, aux, b);

            residualNorm = NormP2(residual);

            logger::Info(headerEilig, "JACOBI iteration = %4u | residual = %10.4e | tol = %10.4e", iteration, residualNorm, rtol);

            if (residualNorm < rtol)
            {
                status = EILIG_SUCCESS;
                x = x2;

                return EILIG_SUCCESS;
            }

            x1 = x2;
        }

        status = EILIG_NOT_CONVERGED;
        logger::Error(headerEilig, "JACOBI solver NOT converged after %d iterations", max);

        return status;
    }
    Status IterativeGauss(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol, Index itmax)
    {
        Status status = EILIG_RUNNING;
        Index numberRows = A.GetRows();
        Index numberCols = A.GetCols();
        Index max = static_cast<Index>(std::pow(numberRows, 1.5));
        Index iteration = { 0 };

        Scalar residualNorm;

        Vector x1(numberRows);
        Vector x2(numberRows);
        Vector u(numberRows);
        Vector v(numberRows);
        Vector aux(numberRows);
        Vector residual(numberRows);
        Ellpack M(numberRows, numberCols);
        Ellpack N(numberRows, numberCols);

        if (itmax != 0)
        {
            max = itmax;
        }

        if (rtol <= 0.)
        {
            status = EILIG_INVALID_TOLERANCE;
            logger::Error(headerEilig, "Convergence tolerance must be a positive real number");
            return status;
        }

        x1 = 0.;
        M = A.Lower(true);
        N = -(A.Upper(false));

        for (iteration = 1; iteration < max; ++iteration)
        {
            Mul(aux, N, x1);
            Add(u, aux, b);
            ForwardLinearSystem(x2, M, u);

            Mul(aux, A, x2);
            Sub(residual, aux, b);

            residualNorm = NormP2(residual);

            logger::Info(headerEilig, "GAUSS iteration = %4u | residual = %10.4e | tol = %10.4e", iteration, residualNorm, rtol);

            if (residualNorm < rtol)
            {
                status = EILIG_SUCCESS;
                x = x2;

                return status;
            }

            x1 = x2;
        }

        status = EILIG_NOT_CONVERGED;
        logger::Error(headerEilig, "GAUSS solver NOT converged after %d iterations", max);

        return status;
    }
    Status IterativeCG(Vector& x, const Ellpack& A, const Vector& b, Scalar rtol, Index itmax)
    {
        Status status = EILIG_RUNNING;
        Scalar residualNorm;
        Scalar alpha;
        Scalar beta;

        Index numberRows = A.GetRows();
        Index max = static_cast<Index>(std::pow(numberRows, 1.5));
        Index iteration = { 0 };

        Vector x0(numberRows);
        Vector r0(numberRows);
        Vector p0(numberRows);
        Vector x1(numberRows);
        Vector r1(numberRows);
        Vector p1(numberRows);
        Vector aux(numberRows);
        Vector residual(numberRows);

        if (itmax != 0)
        {
            max = itmax;
        }

        if (rtol <= 0.)
        {
            status = EILIG_INVALID_TOLERANCE;
            logger::Error(headerEilig, "Convergence tolerance must be a positive real number");
            return status;
        }

        x0 = 0.;

        Mul(aux, -A, x0);
        Add(r0, aux, b);
        p0 = r0;

        for (iteration = 0; iteration < max; ++iteration)
        {
            Mul(aux, A, p0);
            alpha = Dot(r0, r0) / Dot(p0, aux);

            r1 = aux * (-alpha) + r0;
            x1 = p0 * alpha + x0;

            residualNorm = NormP2(r1);

            logger::Info(headerEilig, "CGM iteration = %4u | residual = %10.4e | tol = %10.4e", iteration, residualNorm, rtol);

            if (residualNorm < rtol)
            {
                status = EILIG_SUCCESS;
                x = x1;
                break;
            }

            beta = Dot(r1, r1) / Dot(r0, r0);

            p0 = p0 * beta + r1;

            x0 = x1;
            r0 = r1;
        }

        status = EILIG_NOT_CONVERGED;
        logger::Error(headerEilig, "CGM solver NOT converged after %d iterations", max);

        return status;
    }

    Status IterativeCG(opencl::Vector& x, const opencl::Ellpack& A, const opencl::Vector& b, Scalar rtol, Index itmax)
    {
        Status status = EILIG_RUNNING;
        Scalar residualNorm;
        Scalar alpha;
        Scalar beta;

        Index numberRows = A.GetRows();
        Index max = static_cast<Index>(std::pow(numberRows, 1.5));
        Index iteration = { 0 };

        opencl::Vector x0(b.GetKernels(), numberRows);
        opencl::Vector r0(b.GetKernels(), numberRows);
        opencl::Vector p0(b.GetKernels(), numberRows);
        opencl::Vector x1(b.GetKernels(), numberRows);
        opencl::Vector r1(b.GetKernels(), numberRows);
        opencl::Vector p1(b.GetKernels(), numberRows);
        opencl::Vector aux(b.GetKernels(), numberRows);
        opencl::Vector residual(b.GetKernels(), numberRows);

        if (itmax != 0)
        {
            max = itmax;
        }

        if (rtol <= 0.)
        {
            status = EILIG_INVALID_TOLERANCE;
            logger::Error(headerEilig, "Convergence tolerance must be a positive real number");
            return status;
        }

        x0 = 0.;

        Mul(aux, -A, x0);
        Add(r0, aux, b);
        p0 = r0;

        status = EILIG_NOT_CONVERGED;

        for (iteration = 0; iteration < max; ++iteration)
        {
            Mul(aux, A, p0);
            alpha = Dot(r0, r0) / Dot(p0, aux);

            r1 = aux * (-alpha) + r0;
            x1 = p0 * alpha + x0;

            residualNorm = NormP2(r1);

            logger::Info(headerEilig, "CGM iteration = %4u | residual = %10.4e | tol = %10.4e", iteration, residualNorm, rtol);

            if (residualNorm < rtol)
            {
                status = EILIG_SUCCESS;
                x = x1;
                break;
            }

            beta = Dot(r1, r1) / Dot(r0, r0);

            p0 = p0 * beta + r1;

            x0 = x1;
            r0 = r1;
        }

        if (status != EILIG_SUCCESS)
        {
            logger::Error(headerEilig, "CGM solver NOT converged after %d iterations", max);
            return status;
        }

        return status;
    }
*/