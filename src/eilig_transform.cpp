#include "eilig_transform.hpp"

namespace eilig
{
	namespace transform
	{
		Matrix RotationMatrix(const Vector& axis, Scalar radians)
		{
			Matrix rot(3, 3);

			rot(0, 0) = axis(0) * axis(0) * (1 - cos(radians)) + cos(radians);
			rot(0, 1) = axis(0) * axis(1) * (1 - cos(radians)) - axis(2) * sin(radians);
			rot(0, 2) = axis(0) * axis(2) * (1 - cos(radians)) + axis(1) * sin(radians);
			
			rot(1, 0) = axis(0) * axis(1) * (1 - cos(radians)) + axis(2) * sin(radians);
			rot(1, 1) = axis(1) * axis(1) * (1 - cos(radians)) + cos(radians);
			rot(1, 2) = axis(1) * axis(2) * (1 - cos(radians)) - axis(0) * sin(radians);

			rot(2, 0) = axis(0) * axis(2) * (1 - cos(radians)) - axis(1) * sin(radians);
			rot(2, 1) = axis(1) * axis(2) * (1 - cos(radians)) + axis(0) * sin(radians);
			rot(2, 2) = axis(2) * axis(2) * (1 - cos(radians)) + cos(radians);

			return rot;
		}
		Matrix RotationMatrix(Axis axis, Scalar radians)
		{
			Matrix rot(3, 3, matrix_zeros);

			switch (axis)
			{
			case axis_x:
				rot(0, 0) = 1.0;

				rot(1, 1) = cos(radians);
				rot(1, 2) = -sin(radians);

				rot(2, 1) = sin(radians);
				rot(2, 2) = cos(radians);
				break;
			case axis_y:
				rot(0, 0) = cos(radians);
				rot(0, 2) = sin(radians);

				rot(1, 1) = 1.0;

				rot(2, 0) = -sin(radians);
				rot(2, 2) = cos(radians);
				break;
			case axis_z:
				rot(0, 0) = cos(radians);
				rot(0, 1) = -sin(radians);

				rot(1, 0) = sin(radians);
				rot(1, 1) = cos(radians);

				rot(2, 2) = 1.0;
				break;
			}

			return rot;
		}
		Vector Rotate(const Vector& point, const Vector& axis, Scalar radians)
		{
			return RotationMatrix(axis, radians) * point;
		}
		Vector Rotate(const Vector& point, Axis axis, Scalar radians)
		{
			return RotationMatrix(axis, radians) * point;
		}
		Vector Translate(const Vector& point, Axis axis, Scalar value)
		{
			Vector output(point);

			output(axis) = point(axis) + value;

			return output;
		}
		Vector Mirror(const Vector& point, Axis axis)
		{
			Vector output(point);

			output(axis) = -point(axis);

			return output;
		}
		Vector Scale(const Vector& point, Scalar value)
		{
			Vector output;

			output = value * point;

			return output;
		}
		Matrix TablePointsRotate(const Matrix& input, Axis axis, Scalar radians)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);
			Matrix rot = RotationMatrix(axis, radians);
			Vector vec(3, 0.0);

			if (numberCols < 4)
			{
				logger::Error(headerEilig, "Wrong number of columns.");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				vec(0) = input(i, 0);
				vec(1) = input(i, 1);
				vec(2) = input(i, 2);

				vec = rot * vec;

				output(i, 0) = vec(0);
				output(i, 1) = vec(1);
				output(i, 2) = vec(2);
			}

			return output;
		}
		Matrix TablePointsTranslate(const Matrix& input, Axis axis, Scalar value)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);
			Vector point(3, 0.0);
			Vector aux(3, 0.0);

			if (numberCols < 4)
			{
				logger::Error(headerEilig, "Wrong number of columns.");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				point(0) = input(i, 0);
				point(1) = input(i, 1);
				point(2) = input(i, 2);

				aux = Translate(point, axis, value);

				output(i, 0) = aux(0);
				output(i, 1) = aux(1);
				output(i, 2) = aux(2);
			}

			return output;
		}
		Matrix TablePointsMirror(const Matrix& input, Axis axis)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);
			Vector point(3, 0.0);
			Vector aux(3, 0.0);

			if (numberCols < 4)
			{
				logger::Error(headerEilig, "Wrong number of columns.");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				point(0) = input(i, 0);
				point(1) = input(i, 1);
				point(2) = input(i, 2);

				aux = Mirror(point, axis);

				output(i, 0) = aux(0);
				output(i, 1) = aux(1);
				output(i, 2) = aux(2);
			}

			return output;
		}
		Matrix TablePointsScale(const Matrix& input, Scalar value)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);
			Vector point(3, 0.0);
			Vector aux(3, 0.0);

			if (numberCols < 4)
			{
				logger::Error(headerEilig, "Wrong number of columns.");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				point(0) = input(i, 0);
				point(1) = input(i, 1);
				point(2) = input(i, 2);

				aux = Scale(point, value);

				output(i, 0) = aux(0);
				output(i, 1) = aux(1);
				output(i, 2) = aux(2);
			}

			return output;
		}
		Matrix TableValuesScale(const Matrix& input, Scalar value)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);

			if (numberCols < 4)
			{
				logger::Error(headerEilig, "Wrong number of columns.");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				for (Index j = 3; j < numberCols; ++j)
				{
					output(i, j) = input(i, j) * value;
				}
			}

			return output;
		}
		Matrix TableValuesAdd(const Matrix& input, Scalar value)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);

			if (numberCols < 4)
			{
				logger::Error(headerEilig, "Wrong number of columns.");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				for (Index j = 3; j < numberCols; ++j)
				{
					output(i, j) = input(i, j) + value;
				}
			}

			return output;
		}
		Matrix TableValuesClipBiggerThan(const Matrix& input, Scalar value, Index col)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);

			if (!(col < numberCols))
			{
				logger::Error(headerEilig, "Invalid column for cliping");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				if (output(i, col) > value)
				{
					output(i, col) = value;
				}

			}

			return output;
		}
		Matrix TableValuesClipSmallerThan(const Matrix& input, Scalar value, Index col)
		{
			NumberRows numberRows = input.GetRows();
			NumberCols numberCols = input.GetCols();
			Matrix output(input);

			if (!(col < numberCols))
			{
				logger::Error(headerEilig, "Invalid column for cliping");
				return output;
			}

			for (Index i = 0; i < numberRows; ++i)
			{
				if (output(i, col) < value)
				{
					output(i, col) = value;
				}

			}

			return output;
		}
		Matrix TableAppend(const Matrix& input1, const Matrix& input2)
		{
			NumberRows numberRows1 = input1.GetRows();
			NumberCols numberCols1 = input1.GetCols();
			NumberRows numberRows2 = input2.GetRows();
			NumberCols numberCols2 = input2.GetCols();
			Matrix output;

			if (numberCols1 != numberCols2)
			{
				logger::Error(headerEilig, "Incompatible number of cols.");
				return output;
			}

			output.Resize(numberRows1 + numberRows2, numberCols1);

			for (Index i = 0; i < numberRows1; ++i)
			{
				for (Index j = 0; j < numberCols1; ++j)
				{
					output(i, j) = input1(i, j);
				}
			}

			for (Index i = 0; i < numberRows2; ++i)
			{
				for (Index j = 0; j < numberCols2; ++j)
				{
					output(i + numberRows1, j) = input2(i, j);
				}
			}

			return output;
		}
	}
}