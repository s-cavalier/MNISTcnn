#ifndef CMATRIX
#define CMATRIX
#include "Convolutions.h"
#include "../Dependencies/eigen/Eigen/Dense"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

class CMatrixd {
    std::vector<std::vector<Eigen::MatrixXd>> mat;
    size_t r_size;
    size_t c_size;
    size_t t_size;

public:
    size_t rows() const;
    size_t cols() const;
    size_t size() const;

    static CMatrixd random(const int &r, const int &c, const int &inner_r, const int &inner_c);
    static CMatrixd zero(const int &r, const int &c, const int &inner_r, const int &inner_c);

    bool empty() const;

    //Empty constructor
    CMatrixd();

    //Normal constructor
    CMatrixd(const int& i, const int &j);

    //Copy constructor
    CMatrixd(const CMatrixd &other);

    // Begin iterator
    std::vector<std::vector<Eigen::MatrixXd>>::iterator begin();

    // Returns underlying data. Should be treated very carefully.
    std::vector<std::vector<Eigen::MatrixXd>>& data();

    // Basic no-exception access operator.
    // Default assumes matrix is a vector
    Eigen::MatrixXd& operator()(const int& i, const int &j = 0);

    // Basic no-exception access operator.
    // Default assumes matrix is a vector
    const Eigen::MatrixXd& operator()(const int& i, const int &j = 0) const;

    // Basic access operator.
    Eigen::MatrixXd& at(const int& i, const int &j);

    const Eigen::MatrixXd& at(const int& i, const int &j) const;

    // Assignment operator.
    void operator=(const CMatrixd &other);

    // All operations are done akin to conventional linear algebra
    CMatrixd operator +(const CMatrixd &other) const;
    void operator +=(const CMatrixd &other);
    CMatrixd operator -() const;
    CMatrixd operator -(const CMatrixd &other) const;
    void operator -=(const CMatrixd &other);
    CMatrixd operator *(const double &other) const;
    void operator *=(const double &other);
    CMatrixd operator /(const double &other) const;
    void operator /=(const double &other);

    // Component-wise multiplication
    CMatrixd operator *(const CMatrixd &other) const;
    void operator *=(const CMatrixd &other);

    // Operation order is the same as linear algebra, replace dot products with the sum of convolutions

    // Currently only defined for Matrix/Vector products, could be expanded upon
    CMatrixd convolve(const CMatrixd &other, const bool &full = false) const;
    CMatrixd correlate(const CMatrixd &other, const bool &full = false) const;

    // Exception handling
    // Sins of the father: I hate Eigen error handling
    class invalid_size : public std::runtime_error {
        int i, j, k, l;

    public:
        invalid_size(const std::string &what, const int& lhrows, const int& lhcols, const int& rhrows, const int& rhcols) 
        : std::runtime_error(what), i(lhrows), j(lhcols), k(rhrows), l(rhcols) { std::cerr << what << std::endl; };

        const int& lefthand_rows() const;
        const int& lefthand_cols() const;
        const int& righthand_rows() const;
        const int& righthand_cols() const;
    };

};



#endif