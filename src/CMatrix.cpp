#include "CMatrix.h"
#include <iostream>

const int& CMatrixd::invalid_size::lefthand_rows() const { return i; }
const int& CMatrixd::invalid_size::lefthand_cols() const { return j; }
const int& CMatrixd::invalid_size::righthand_rows() const { return k; }
const int& CMatrixd::invalid_size::righthand_cols() const { return l; }


CMatrixd::CMatrixd() {
    r_size = 0;
    c_size = 0;
    t_size = 0;
}

CMatrixd::CMatrixd(const int &i, const int& j) {
    mat.resize( i, std::vector<Eigen::MatrixXd>(j) );
    r_size = i;
    c_size = j;
    t_size = i * j;
}

CMatrixd::CMatrixd(const CMatrixd &other) {
    mat.resize( other.rows(), std::vector<Eigen::MatrixXd>(other.cols()) );
    for (int i = 0; i < other.rows(); i++) {
        for (int j = 0; j < other.cols(); j++) mat[i][j] = other(i, j);
    }
    r_size = other.rows();
    c_size = other.cols();
    t_size = r_size * c_size;
}

size_t CMatrixd::rows() const {
    return r_size;
}

size_t CMatrixd::cols() const {
    return c_size;
}

size_t CMatrixd::size() const {
    return t_size;
}

std::vector<std::vector<Eigen::MatrixXd>>& CMatrixd::data() {
    return mat;
}

Eigen::MatrixXd& CMatrixd::operator()(const int& i, const int &j) {
    return mat[i][j];
}

const Eigen::MatrixXd& CMatrixd::operator()(const int& i, const int &j) const {
    return mat[i][j];
}

Eigen::MatrixXd& CMatrixd::at(const int& i, const int &j) {
    return mat.at(i).at(j);
}

const Eigen::MatrixXd& CMatrixd::at(const int& i, const int &j) const {
    return mat.at(i).at(j);
}


void CMatrixd::operator=(const CMatrixd &other) {
    mat.resize(other.rows(), std::vector<Eigen::MatrixXd>(other.cols()));

    r_size = other.rows();
    c_size = other.cols();
    t_size = r_size * c_size;

    for (int i = 0; i < other.rows(); i++) {
        for (int j = 0; j < other.cols(); j++) mat[i][j] = other(i, j);
    }
}

CMatrixd CMatrixd::operator +(const CMatrixd &other) const {
    if (rows() != other.rows() || cols() != other.cols() ) throw invalid_size("Dimensions of two matrices must match for an addition operation.", rows(), cols(), other.rows(), other.cols());
    CMatrixd out(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) out(i, j) = mat[i][j] + other(i, j);
    }
    return out;
}

void CMatrixd::operator +=(const CMatrixd &other) {
    if (rows() != other.rows() || cols() != other.cols() ) throw invalid_size("Dimensions of two matrices must match for an addition operation.", rows(), cols(), other.rows(), other.cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) mat.at(i).at(j) += other(i, j);
    }
}

CMatrixd CMatrixd::operator -() const {
    CMatrixd out(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) out(i, j) = -mat[i][j];
    }
    return out;
}

CMatrixd CMatrixd::operator -(const CMatrixd &other) const {
    if (rows() != other.rows() || cols() != other.cols() ) throw invalid_size("Dimensions of two matrices must match for a subtraction operation.", rows(), cols(), other.rows(), other.cols());
    CMatrixd out(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) out(i, j) = mat[i][j] - other(i, j);
    }
    return out;
}

void CMatrixd::operator -=(const CMatrixd &other) {
    if (rows() != other.rows() || cols() != other.cols() ) throw invalid_size("Dimensions of two matrices must match for a subtraction operation.", rows(), cols(), other.rows(), other.cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) mat[i][j] -= other(i, j);
    }
}

CMatrixd CMatrixd::operator *(const double &other) const {
    CMatrixd out(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) out(i, j) = mat[i][j] * other;
    }
    return out;
}

void CMatrixd::operator *=(const double &other) {
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) mat[i][j] *= other;
    }
}

CMatrixd CMatrixd::operator /(const double &other) const {
    return *this * (1.0 / other);
}

void CMatrixd::operator /=(const double &other) {
    return *this *= (1.0 / other);
}

CMatrixd CMatrixd::convolve(const CMatrixd &other, const bool &full) const {
    if (other.cols() != 1 || cols() != other.rows()) throw ("Columns must match rows of input matrix", rows(), cols(), other.rows(), other.cols());
    CMatrixd out(rows(), 1);
    for (int i = 0; i < rows(); i++) {
        int rdims;
        int cdims;

        if (full) {
            rdims = mat[0][i].rows() + other(i).rows() - 1;
            cdims = mat[0][i].cols() + other(i).cols() - 1;
        } else {
            rdims = mat[0][i].rows() - other(i).rows() + 1;
            cdims = mat[0][i].cols() - other(i).cols() + 1;
        }

        out(i) = Eigen::MatrixXd::Zero(rdims, cdims);
        for (int j = 0; j < cols(); j++) out(i) += Convolutions::fftconvolve2d( other(j), mat[i][j], full);
    }

    return out;
}

CMatrixd CMatrixd::correlate(const CMatrixd &other, const bool &full) const {
    if (other.cols() != 1 || cols() != other.rows()) throw ("Columns must match rows of input matrix", rows(), cols(), other.rows(), other.cols());
    CMatrixd out(rows(), 1);
    for (int i = 0; i < rows(); i++) {
        out(i) = Convolutions::fftcorrelate2d( other(0), mat[i][0], full);
        for (int j = 1; j < cols(); j++) {
            out(i) += Convolutions::fftcorrelate2d( other(j), mat[i][j], full);
        }
    }

    return out;
}

CMatrixd CMatrixd::operator *(const CMatrixd &other) const {
    CMatrixd out(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) out(i, j) = mat[i][j].cwiseProduct(other(i, j) );
    }
    return out;
}

void CMatrixd::operator *=(const CMatrixd &other) {
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) mat[i][j] = mat[i][j].cwiseProduct(other(i, j) );
    }
}


CMatrixd CMatrixd::random(const int &i, const int &j, const int &inner_r, const int &inner_c) {
    CMatrixd out(i, j);
    for (int i = 0; i < out.rows(); i++) {
        for (int j = 0; j < out.cols(); j++) out(i, j) = Eigen::MatrixXd::Random(inner_r, inner_c);
    }
    return out;
}

CMatrixd CMatrixd::zero(const int &i, const int &j, const int &inner_r, const int &inner_c) {
    CMatrixd out(i, j);
    for (int i = 0; i < out.rows(); i++) {
        for (int j = 0; j < out.cols(); j++) out(i, j) = Eigen::MatrixXd::Zero(inner_r, inner_c);
    }
    return out;
}

bool CMatrixd::empty() const {
    return size() == 0;
}

std::vector<std::vector<Eigen::MatrixXd>>::iterator CMatrixd::begin() {
    return mat.begin();
}