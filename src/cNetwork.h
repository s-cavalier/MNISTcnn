#ifndef CN_NETWORK
#define CN_NETWORK
#include "CMatrix.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <deque>
#include <utility>
#include <algorithm>
#include <cmath>

class cNetwork {

    // Used for the max pooling layers to hold locations of the max values
    // Rounds the double before hashing
    template <int error> struct doubleHasher {
        size_t operator()(const double &x) const {
            return std::hash<double>()(  (double)((int)(x * error)) / error );
        }
    };

    // Used for the max pooling layers to hold locations of the max values
    // Attempts approximate equivalence rather than exact equivalence
    template <int error> struct doubleEqual {
        bool operator()(const double &l, const double &r) const {
            return abs(l - r) < (1.0 / error);
        }
    };

    int input_size, clayers, layers, output_nodes;

    //Sigmoid function
    static double sig(const double &f);

    //Component-wise sigmoid function. Operates on the vector inplace.
    static void sig_v(Eigen::VectorXd &v);

    //Component-wise sigmoid function. Operates on the matrix inplace.
    static void sig_m(Eigen::MatrixXd &m);

    //Component-wise sigmoid function. Operates on the matrix inplace.
    static void sig_cm(CMatrixd &m);

    //Componenet-wise sigmoid prime function. Returns copy of vector, does not modify original.
    static Eigen::MatrixXd sig_prime(const Eigen::MatrixXd &v);

    static long double large_power_of_two(long double x);
    static long double large_root_of_two(long double x);

public:
    struct gradient_set {
        std::deque<CMatrixd> kernels_gradient;
        std::deque<CMatrixd> cbiases_gradient;
        std::deque<Eigen::MatrixXd> weights_gradient;
        std::deque<Eigen::VectorXd> biases_gradient;
    };

    //returns weights & bias gradients
    //first = weights, second = biases
    gradient_set backpropagate(const Eigen::MatrixXd& input, const Eigen::VectorXd& exp_output, std::vector<Eigen::VectorXd>* outputs = nullptr);

    double learning_rate = 0.15;

    using mstorage = std::unordered_map<double, std::pair<int, int>, doubleHasher<1000>, doubleEqual<1000>>;
    // Computes a max pool of size & stride with O(1) space complexity and O(n) time complexity
    // Optimizations could be made by optimizing root function
    static Eigen::MatrixXd max_pool2d(const Eigen::MatrixXd& m, const int& size, const int& stride, mstorage* maxes = nullptr);

    //Convolutional weights
    std::vector<CMatrixd> kernels;
    //Convolutional biases
    std::vector<CMatrixd> cbiases;

    // Normal NN weights
    std::vector<Eigen::MatrixXd> weights;
    // Normal NN biases
    std::vector<Eigen::VectorXd> biases;

    //Max pool information
    std::unordered_map<int, std::pair<int, int>> max_pool_layers;

    

    // Defines the structure of the network
    // input should be a matrix of input_size x input_size, should be a square matrix
    // max_pooling_layers will track which layers defined by the next parameter should be max pooling layers 
    // Kernels_sizes defines the structure of the CNN; first defines the num of kernels (the dim of the next layer), second defines the width & height of each layer (each kernel is a square kernel)
    // If defined by the previous parameter to be a max pooling layer, first = stride, second = size.
    // node_sizes defines the structure of the NN; each is the nodes in a given layer
    // output_nodes defineds the vector that should be output
    cNetwork(const int& input_size, const std::unordered_set<int>& max_pooling_layers, const std::vector<std::pair<int, int>>& kernels_sizes, std::vector<int> node_sizes, const int& output_nodes);

    // Does a feedforward and returns the FF output.
    Eigen::VectorXd activate(const Eigen::MatrixXd& input) const;

    // Does a backpropagation with mini-batch size defined to be the size of the input
    // A pointer to an output vector is allowed to track outputs
    void train(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::VectorXd>& exp_outputs, std::vector<Eigen::VectorXd>* outputs = nullptr);

};


#endif