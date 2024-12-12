#ifndef EILIG_TRANSFORM_HPP_
#define EILIG_TRANSFORM_HPP_

#include "eilig_types.hpp"
#include "eilig_matrix.hpp"
#include "eilig_vector.hpp"
#include "eilig_routines.hpp"

namespace eilig
{
	namespace transform
	{
		Matrix RotationMatrix(const Vector& axis, Scalar radians);
		Matrix RotationMatrix(Axis axis, Scalar radians);
		Vector Rotate(const Vector& point, const Vector& axis, Scalar radians);
		Vector Rotate(const Vector& point, Axis axis, Scalar radians);
		Vector Translate(const Vector& point, Axis axis, Scalar value);
		Vector Mirror(const Vector& point, Axis axis);
		Vector Scale(const Vector& point, Scalar value);

		Matrix TablePointsRotate(const Matrix& input, Axis axis, Scalar radians);
		Matrix TablePointsTranslate(const Matrix& input, Axis axis, Scalar value);
		Matrix TablePointsMirror(const Matrix& input, Axis axis);
		Matrix TablePointsScale(const Matrix& input, Scalar value);
		
		Matrix TableValuesScale(const Matrix& input, Scalar value);
		Matrix TableValuesAdd(const Matrix& input, Scalar value);
		Matrix TableValuesClipBiggerThan(const Matrix& input, Scalar value, Index col);
		Matrix TableValuesClipSmallerThan(const Matrix& input, Scalar value, Index col);
		
		Matrix TableAppend(const Matrix& input1, const Matrix& input2);
	}
}

#endif /* EILIGt_TRANSFORM_HPP_ */