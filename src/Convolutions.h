#ifndef CONVLS
#define CONVLS
#include "../Dependencies/eigen/Eigen/Dense"

class Convolutions {

    static inline const double PI = 3.14159265359f;

    // Inverse Fast Fourier Transform, requires zero-padding to the nearest power of two and multiplied by 1 / v.size() after completion
    static Eigen::VectorXcd innerifft(const Eigen::VectorXcd &v);

public:

    // Rounds up to nearest power of two
    static unsigned long nearest_power(unsigned long v);

    // Fast Fourier Transform, requires zero-padding to nearest power of two
    static Eigen::VectorXcd fft(const Eigen::VectorXcd& v);

    // Inverse Fast Fourier Transform, requires zero-padding to nearest power of two
    static Eigen::VectorXcd ifft(const Eigen::VectorXcd &v);

    // Fast Fourier Transform Convolution
    static Eigen::VectorXd fftconvolve(const Eigen::VectorXd &A, const Eigen::VectorXd &B);

    // 2d fft, applies col-wise then row-ise fft, requires 0 padding (both row len and col len should be a power of 2)
    static Eigen::MatrixXcd fft2d(const Eigen::MatrixXcd &m);

    // 2d fft, applies col-wise then row-wise fft, requires 0 padding (both row len and col len should be a power of 2)
    static Eigen::MatrixXcd ifft2d(const Eigen::MatrixXcd &m);

    // FFT Convolution for 2d vectors (matrices)
    // full = false will provide a "valid" convolution beginning where all the elements of the kernel are "within" X
    static Eigen::MatrixXd fftconvolve2d(const Eigen::MatrixXd& X, const Eigen::MatrixXd &kernel, bool full = false);

    // FFT Cross correlation for 2d vectors (matrices)
    // full = false will provide a "valid" convolution beginning where all the elements of the kernel are "within" X
    static Eigen::MatrixXd fftcorrelate2d(const Eigen::MatrixXd& X, Eigen::MatrixXd kernel, bool full = false);


};

#endif