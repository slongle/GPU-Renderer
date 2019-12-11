#include "transform.h"

Matrix4x4::Matrix4x4(const Float m[16])
{
	for (int i = 0; i < 16; i++){
		m_m[i] = m[i];
	}
}

void Matrix4x4::Identity()
{
	for (int i = 0; i < 16; i++) {
		m_m[i] = 0;
	}
	for (int i = 0; i < 4; i++){
		m_m[i * 4 + i] = 1;
	}
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4& t) const
{
	Float m[16];
	for (int i = 0; i < 4; i++)	{
		for (int j = 0; j < 4; j++)	{
			m[i * 4 + j] = 0;
			for (int k = 0; k < 4; k++)	{
				m[i * 4 + j]+=this->m_m[i*4+k]*t.m_m[k*4+j];	
			}
		}
	}
	return Matrix4x4(m);
}

Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& t)
{
	Float m[16];
	for (int i = 0; i < 4; i++)	{
		for (int j = 0; j < 4; j++)	{
			m[i * 4 + j] = 0;
			for (int k = 0; k < 4; k++)	{
				m[i * 4 + j] += this->m_m[i * 4 + k] * t.m_m[k * 4 + j];
			}
		}
	}
	for (int i = 0; i < 16; i++){
		this->m_m[i] = m[i];
	}
	return *this;
}

Matrix4x4 Inverse(const Matrix4x4& t)
{
	Float m[4][4];
	Float mInv[4][4];
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++)	{
			m[i][j] = t.m_m[i * 4 + j];
		}
	}
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mInv[i][j] = 0;
		}
	}
	for (int i = 0; i < 4; i++) {
		mInv[i][i] = 1;
	}
	int singularFlag = 0;
	for (int i = 0; i < 3; i++){
		if (m[i][i] == 0)	{
			singularFlag = 1;
			for (int j = i + 1; j < 4; j++)	{
				if (m[j][i] != 0){
					for (int k = i; k < 4; k++)	{
						m[i][k] += m[j][k];
					}
					singularFlag = 0;
					break;
				}
			}
			ASSERT(singularFlag == 0, "Martrix is singular!");
		}
		for (int j = i+1; j < 4; j++){
			Float ratio =- m[j][i]/m[i][i];
			for (int k = i; k < 4; k++) {
				m[j][k] += m[i][k] * ratio;
			}
			for (int k = 0; k < 4; k++) {
				mInv[j][k] += mInv[i][k] * ratio;
			}
		}
	}
	ASSERT(m[3][3] == 0, "Martrix is singular!");
	for (int i = 3; i > 0; i--) {
		Float ratio = 1 / m[i][i];
		for (int k = i; k < 4; k++){
			m[i][k] *= ratio;
		}
		for (int k = 0; k < 4; k++) {
			mInv[i][k] *= ratio;
		}
		for (int j = i - 1; j >= 0; j--) {
			ratio = -m[j][i];
			for (int k = i; k < 4; k++) {
				m[j][k] += m[i][k] * ratio;
			}
			for (int k = 0; k < 4; k++) {
				mInv[j][k] += mInv[i][k] * ratio;
			}
		}
	}
	return Matrix4x4((Float*)mInv);
}



Transform::Transform(const Float m[16]) :m_mat(m), m_invMat(Inverse(m_mat)) {}

Transform::Transform(const Matrix4x4 m) :m_mat(m), m_invMat(Inverse(m_mat)) {}

Transform::Transform(const Float m[16], const Float mInv[16]) :m_mat(m), m_invMat(mInv) {}

Transform::Transform(const Matrix4x4 m, const Matrix4x4 mInv) :m_mat(m), m_invMat(mInv) {}

void Transform::Identity() 
{
	m_mat.Identity();
	m_invMat.Identity();
}

Transform Inverse(const Transform& t)
{
	return Transform(t.m_invMat,t.m_mat);
}

Transform Transform::operator*(const Transform& t) const
{
	return Transform(m_mat*t.m_mat, t.m_invMat*m_invMat);
}

Transform& Transform::operator*=(const Transform& t)
{
	m_mat *= t.m_mat;
	m_invMat = t.m_invMat * m_invMat;
	return *this;
}
