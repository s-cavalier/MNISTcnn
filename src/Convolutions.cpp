#include "Convolutions.h"
#include <complex>
#include <iostream>
using namespace std::complex_literals;

unsigned long Convolutions::nearest_power(unsigned long v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

Eigen::VectorXcd Convolutions::fft(const Eigen::VectorXcd& coefficients) {
    // Base case
    if (coefficients.size() <= 1) return coefficients;
    int n = coefficients.size();
    // Roots of unity
    std::complex<double> omega = exp( 2.0i * PI / (double)n);
    Eigen::VectorXcd Peven(n / 2), Podd(n / 2);
    // Split into even and odd polynomials
    int at = 0;
    for (int i = 0; i < n - 1; i += 2) Peven(at++) = coefficients(i);
    at = 0;
    for (int i = 1; i < n; i += 2) Podd(at++) = coefficients(i);
    Peven = fft(Peven);
    Podd = fft(Podd);
    Eigen::VectorXcd out(n);
    for (int i = 0; i < n / 2; i++) {
        // Set positive/negative pairs based on roots of unity
        out(i) = Peven(i) + pow(omega, i) * Podd(i);
        out(i + n / 2) = Peven(i) - pow(omega, i) * Podd(i);
    }
    return out;
}

Eigen::VectorXcd Convolutions::innerifft(const Eigen::VectorXcd& coefficients) {
    // Base case
    if (coefficients.size() <= 1) return coefficients;
    int n = coefficients.size();
    // Roots of unity (in opposite direction per inverse matrix)
    std::complex<double> omega = exp(-2.0i * PI / (double)n);
    Eigen::VectorXcd Peven(n / 2), Podd(n / 2);
    // Split into even and odd polynomials
    int at = 0;
    for (int i = 0; i < n - 1; i += 2) Peven(at++) = coefficients(i);
    at = 0;
    for (int i = 1; i < n; i += 2) Podd(at++) = coefficients(i);
    Peven = innerifft(Peven);
    Podd = innerifft(Podd);
    Eigen::VectorXcd out(n);
    for (int i = 0; i < n / 2; i++) {
        // Set positive / negative pairs based on roots of unity
        out(i) = Peven(i) + pow(omega, i) * Podd(i);
        out(i + n / 2) = Peven(i) - pow(omega, i) * Podd(i);
    }
    return out;
}

Eigen::VectorXcd Convolutions::ifft(const Eigen::VectorXcd& coefficients) {
    return innerifft(coefficients) * (1.0 / coefficients.size());
}

Eigen::VectorXd Convolutions::fftconvolve(const Eigen::VectorXd& A, const Eigen::VectorXd& B) {
    // Get size and zero-pad vectors to be a power of 2
    int n = A.size() + B.size();
    int true_n = nearest_power(n);
    Eigen::VectorXcd a = Eigen::VectorXcd::Constant(true_n, 0);
    Eigen::VectorXcd b = Eigen::VectorXcd::Constant(true_n, 0);
    int end = std::max(A.size(), B.size());
    for (int i = 0; i < end; i++) {
        if (i < A.size()) a(i) = A(i);
        if (i < B.size()) b(i) = B(i);
    }
    // Multiply the FFT'd vectors pointwise (hamamard product) and IFFT that vector, return the real part with any padded zeros cut off
    return ifft( fft(a).array() * fft(b).array() ).real().block(0, 0, n - 1, 1);
}

Eigen::MatrixXcd Convolutions::fft2d(const Eigen::MatrixXcd &m) {
    Eigen::MatrixXcd output(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++) output.row(i) = fft(m.row(i));
    for (int i = 0; i < m.cols(); i++) output.col(i) = fft(output.col(i));
    return output;
}

Eigen::MatrixXcd Convolutions::ifft2d(const Eigen::MatrixXcd &m) {
    Eigen::MatrixXcd output(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); i++) output.row(i) = ifft(m.row(i));
    for (int i = 0; i < m.cols(); i++) output.col(i) = ifft(output.col(i));
    return output;
}

// Assumes 0, 0 is the top-left index
Eigen::MatrixXd Convolutions::fftconvolve2d(const Eigen::MatrixXd& X, const Eigen::MatrixXd& kernel, bool full) {

    int n = X.rows() + kernel.rows();
    int m = X.cols() + kernel.cols();
    int true_n = nearest_power(n);
    int true_m = nearest_power(m);
    Eigen::MatrixXcd x = Eigen::MatrixXcd::Constant(true_n, true_m, 0);
    Eigen::MatrixXcd k = Eigen::MatrixXcd::Constant(true_n, true_m, 0);
    int end_r = std::max(X.rows(), kernel.rows());
    int end_c = std::max(X.cols(), kernel.cols()); 
    for (int i = 0; i < end_r; i++) {
        for (int j = 0; j < end_r; j++) {
            if (i < X.rows() && j < X.cols()) x(i, j) = X(i, j);
            if (i < kernel.rows() && j < kernel.cols()) k(i, j) = kernel(i, j);
        }
    }
    if (full) return ifft2d( fft2d( x ).array() * fft2d( k ).array() ).real().block(0, 0, n - 1, m - 1);
    int trim_r = X.rows() - kernel.rows() + 1;
    int trim_c = X.cols() - kernel.cols() + 1;
    return ifft2d( fft2d( x ).array() * fft2d( k ).array() ).real().block((n - trim_r) / 2, (m - trim_c) / 2, trim_r, trim_c);
}

Eigen::MatrixXd Convolutions::fftcorrelate2d(const Eigen::MatrixXd& X, Eigen::MatrixXd kernel, bool full) {
    // switch K for a cross correlation
    kernel.colwise().reverseInPlace();
    kernel.rowwise().reverseInPlace();

    int n = X.rows() + kernel.rows();
    int m = X.cols() + kernel.cols();
    int true_n = nearest_power(n);
    int true_m = nearest_power(m);
    Eigen::MatrixXcd x = Eigen::MatrixXcd::Constant(true_n, true_m, 0);
    Eigen::MatrixXcd k = Eigen::MatrixXcd::Constant(true_n, true_m, 0);
    int end_r = std::max(X.rows(), kernel.rows());
    int end_c = std::max(X.cols(), kernel.cols()); 
    for (int i = 0; i < end_r; i++) {
        for (int j = 0; j < end_r; j++) {
            if (i < X.rows() && j < X.cols()) x(i, j) = X(i, j);
            if (i < kernel.rows() && j < kernel.cols()) k(i, j) = kernel(i, j);
        }
    }
    if (full) return ifft2d( fft2d( x ).array() * fft2d( k ).array() ).real().block(0, 0, n - 1, m - 1);
    int trim_r = X.rows() - kernel.rows() + 1;
    int trim_c = X.cols() - kernel.cols() + 1;
    return ifft2d( fft2d( x ).array() * fft2d( k ).array() ).real().block((n - trim_r) / 2, (m - trim_c) / 2, trim_r, trim_c);
}