#include "cNetwork.h"
#include <stdexcept>
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <string>
#include <unordered_map>
#include <deque>

cNetwork::cNetwork(const int& input_size, const std::unordered_set<int>& max_pool_indicies, const std::vector<std::pair<int, int>>& kernels_sizes, std::vector<int> node_sizes, const int& output_nodes) {
    this->input_size = input_size; 
    this->output_nodes = output_nodes;
    clayers = kernels_sizes.size();
    layers = node_sizes.size() + 1;
    std::srand(time(0));

    // Intitialize Kernels & CBiases
    int prev_len = input_size;
    int prev_dim = 1;
    for (int i = 0; i < clayers; i++) {
        
        // Manage max pooling layers
        if (max_pool_indicies.count(i)) {
            int new_len = (prev_len - kernels_sizes[i].second) / kernels_sizes[i].first + 1;
            max_pool_layers[i] = kernels_sizes[i];
            kernels.push_back( CMatrixd() );
            cbiases.push_back( CMatrixd::zero(prev_dim, 1, new_len, new_len) );
            prev_len = new_len;
            continue;
        }

        // Manage normal convolutional layers
        int new_len = prev_len - kernels_sizes[i].second + 1;
        kernels.push_back( CMatrixd::random(kernels_sizes[i].first, prev_dim, kernels_sizes[i].second, kernels_sizes[i].second) );
        cbiases.push_back( CMatrixd::random(kernels_sizes[i].first, 1, new_len, new_len) );
        prev_len = new_len;
        prev_dim = kernels_sizes[i].first;
    }

    // Normal layers after converting final Clayer into nodal layer
    prev_len *= prev_len * prev_dim;
    node_sizes.push_back(output_nodes);
    for (int i = 0; i < node_sizes.size(); i++) {
        weights.push_back(Eigen::MatrixXd::Random(node_sizes[i], prev_len) * 2);
        biases.push_back(Eigen::VectorXd::Random(node_sizes[i]) * 2);

        prev_len = node_sizes[i];
    }
}

double cNetwork::sig(const double &f) {
    return 1.0 / (1 + std::exp(-f));
}

//Sigmoid function
void cNetwork::sig_v(Eigen::VectorXd &v) {
    for (int i = 0; i < v.rows(); i++) v(i) = sig(v(i));
}

void cNetwork::sig_m(Eigen::MatrixXd &m) {
    for (int r = 0; r < m.rows(); r++) {
        for (int c = 0; c < m.cols(); c++) m(r, c) = sig( m(r, c) );
    }
}

void cNetwork::sig_cm(CMatrixd &m) {
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) sig_m(m(i, j));
    }
}

//d/dx Sigmoid Function
Eigen::MatrixXd cNetwork::sig_prime(const Eigen::MatrixXd &v) {
    Eigen::MatrixXd out = v;
    for (int i = 0; i < v.rows(); i++) {
        for (int j = 0; j < v.cols(); j++) out(i) = sig( out(i) ) * ( 1 - sig( out(i) ) );
    }
    return out;
}

long double cNetwork::large_power_of_two(long double x) {
    x *= x; // ^2
    x *= x; // ^4
    x *= x; // ^8
    x *= x; // ^16
    x *= x; // ^32
    x *= x; // ^64
    x *= x; // ^128
    x *= x; // ^256
    return x;
}


long double cNetwork::large_root_of_two(long double x) {
    return sqrtl(
        sqrtl(
            sqrtl(
                sqrtl(
                    sqrtl(
                        sqrtl(
                            sqrtl(
                                sqrtl(x)
        ))))))); // x ^ (1 / 256)
}


Eigen::MatrixXd cNetwork::max_pool2d(const Eigen::MatrixXd& m, const int& size, const int& stride, mstorage* maxes) {
    
    Eigen::MatrixXd cNetwork::max_pool2d(const Eigen::MatrixXd& m, const int& size, const int& stride, mstorage* maxes) {
    
    // i represents the current "y" position of the upper row
    // j represents the current "x" position of the left column of the box about to be lost
    // k represents the current "x" position of the left column about to be gained
    /*
            lost curr gain
            ____________
        i : j  |    k  |
     size | |__|____|__|
    
    */
   

    int new_size = (m.rows() - size) / stride + 1;
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(new_size, new_size);
    for (int i = 0; i + size <= m.rows(); i += stride) {

        long double sum = 0;
        double max = 0;

        // Initialize "sliding window"
        for (int j = i; j < i + size; j++) {
            for (int k = 0; k < size; k++) {
                sum += large_power_of_two(m(j, k));
                if (m(j, k) <= max) continue;
                max = m(j, k);
            }
        }
        out(i / stride, 0) = max;

        // Start moving window
        for (int j = 0, k = size; k + stride <= m.cols(); j += stride, k += stride) {


            // Box of lost stuff
            for (int l = i; l < i + size; l++) {
                for (int x = j; x < j + stride; x++) sum -= large_power_of_two(m(l, x));
            }

            // Account for floating point arithmetic errors (can blow up quickly)
            sum = std::max(sum, 0.0l);

            // Box of new stuff
            for (int l = i; l < i + size; l++) {
                for (int x = k; x < k + stride; x++) sum += large_power_of_two(m(l, x));
            }

            double new_max = large_root_of_two(sum);
            int ox = i / stride;
            int oy = j / stride + 1;
            out(i / stride, j / stride + 1) = new_max;
            if (maxes) (*maxes)[new_max] = { ox, oy };

        }
    }

    return out;    
    
}
    
}

Eigen::VectorXd cNetwork::activate(const Eigen::MatrixXd& input) const {
    CMatrixd active(1, 1);
    active(0) = input;
    for (int i = 0; i < clayers; i++) {
        // Manage max pooling case
        if (max_pool_layers.count(i)) {
            for (int j = 0; j < active.rows(); j++) active(j) = max_pool2d( active(j), max_pool_layers.at(i).second, max_pool_layers.at(i).first );
            continue;
        }

        // Manage normal convolutional layer
        active = kernels[i].correlate(active) + cbiases[i];
        sig_cm(active);
        
    }

    // Convert to vector
    Eigen::VectorXd nodes( active.rows() * active(0).rows() * active(0).cols() );
    int at = 0;
    for (int i = 0; i < active.rows(); i++) {
        for (int j = 0; j < active(i).rows(); j++) {
            for (int k = 0; k < active(i).cols(); k++) nodes(at++) = active(i)(j, k);
        }
    }

    // Normal feedforward
    for (int i = 0; i < layers; i++) {
        nodes = weights[i] * nodes + biases[i];
        sig_v(nodes);
    }

    return nodes;
}

cNetwork::gradient_set cNetwork::backpropagate(const Eigen::MatrixXd& input, const Eigen::VectorXd& exp_output, std::vector<Eigen::VectorXd>* outputs) {
    std::vector<CMatrixd> c_z;
    std::vector<CMatrixd> c_a;
    std::unordered_map<int, mstorage> max_locations;
    //Compute feedforward
    CMatrixd active(1, 1);
    active(0) = input;
    c_a.push_back(active);
    for (int i = 0; i < clayers; i++) {
        // Manage max pooling case
        if (max_pool_layers.count(i)) {
            mstorage maxloc;
            for (int j = 0; j < active.rows(); j++) active(j) = max_pool2d( active(j), max_pool_layers.at(i).second, max_pool_layers.at(i).first, &maxloc );
            max_locations[i] = maxloc;
            c_z.push_back( active );
            c_a.push_back( active );
            continue;
        }

        // Manage normal convolutional layer
        active = kernels[i].correlate(active) + cbiases[i];
        c_z.push_back( active );
        sig_cm(active);
        c_a.push_back( active );
    }

    // Convert to vector
    Eigen::VectorXd nodes( active.rows() * active(0).rows() * active(0).cols() );
    int at = 0;
    for (int i = 0; i < active.rows(); i++) {
        for (int j = 0; j < active(i).rows(); j++) {
            for (int k = 0; k < active(i).cols(); k++) nodes(at++) = active(i)(j, k);
        }
    }

    std::vector<Eigen::VectorXd> z;
    std::vector<Eigen::VectorXd> a;
    a.push_back(nodes);

    // Normal feedforward
    for (int i = 0; i < layers; i++) {
        nodes = weights[i] * nodes + biases[i];
        z.push_back(nodes);
        sig_v(nodes);
        a.push_back(nodes);
    }

    if (outputs) outputs->push_back(nodes);
    gradient_set out;

    // Back propagate
    // Normal NN backprop first
    // Start with delC/delz^L = delC/dela^L * dela^L/delz^L = a^L - y for a log based cost function
    // Confirmed works
    Eigen::VectorXd dCdz = a[layers] - exp_output;
    out.weights_gradient.push_front( dCdz * a[layers - 1].transpose() );
    out.biases_gradient.push_front( dCdz );
    for (int i = layers - 2; i >= 0; i--) {
        dCdz = (weights[i + 1].transpose() * ( dCdz )).array() * sig_prime(z[i]).array();
        out.weights_gradient.push_front( dCdz * a[i].transpose() );
        out.biases_gradient.push_front( dCdz );
    }

    // Convert dCdz into CMatrix of dCdZ's
    dCdz = weights[0].transpose() * dCdz;
    CMatrixd dCdZ(active.rows(), 1);
    for (int i = 0; i < dCdZ.rows(); i++) {
        dCdZ(i).resizeLike( c_z.back()(i) );
        at = 0;
        for (int j = 0; j < dCdZ(i).rows(); j++) {
            for (int k = 0; k < dCdZ(i).cols(); k++) dCdZ(i)(j, k) = dCdz(at++);
        }
    }

    // Core backprop
    // Final layer (dC/dZ^L has been established where L = the final c layer)
    for (int i = clayers - 1; i >= 0; i--) {

        // Manage max pooling case
        if ( max_pool_layers.count(i) ) {
            CMatrixd prev( dCdZ.rows(), 1 );
            for (int j = 0; j < dCdZ.rows(); j++) {
                prev(j).resizeLike( c_a[i](j) );
                // Use the saved sets in max_locations to iterate through and grab locations of where the value went
                for (int k = 0; k < prev(j).rows(); k++) {
                    for (int l = 0; l < prev(j).cols(); l++) {
                        if (!max_locations[i].count( c_a[i](j)(k, l) )) {
                            prev(j)(k, l) = 0;
                            continue;
                        }
                        auto &location = max_locations[i][ c_a[i](j)(k, l) ];
                        prev(j)(k, l) = dCdZ(j)( location.first, location.second );
                    }
                }
            }

            dCdZ = prev;
            continue;
        }

        // NN Cast
        for (int j = 0; j < dCdZ.rows(); j++) {
            dCdZ(j) = dCdZ(j).cwiseProduct(sig_prime( c_z[i](j) ));
        }
        
        CMatrixd dCdK( kernels[i].rows(), kernels[i].cols() );

        // Kernel gradient
        // Possibly replacable by some kind of tranpose argument
        for (int j = 0; j < kernels[i].rows(); j++) {
            for (int k = 0; k < kernels[i].cols(); k++) dCdK(j, k) = Convolutions::fftcorrelate2d( c_a[i](k), dCdZ(j) );
        }

        out.kernels_gradient.push_front( dCdK );

        // Biases gradient
        out.cbiases_gradient.push_front( dCdZ );

        if (i == 0) break;
        CMatrixd prev( c_a[i].rows(), 1 );
        for (int j = 0; j < prev.rows(); j++ ) {
            prev(j) = Convolutions::fftconvolve2d( dCdZ(0), kernels[i](0, j), true );
            for (int k = 1; k < dCdZ.rows(); k++) prev(j) += Convolutions::fftconvolve2d( dCdZ(k), kernels[i](k, j) , true);
        }
        dCdZ = prev;
    }

    return out;
}

void cNetwork::train(const std::vector<Eigen::MatrixXd>& inputs, const std::vector<Eigen::VectorXd>& exp_outputs, std::vector<Eigen::VectorXd>* outputs) {
    // Establish baseline
    gradient_set gradients = backpropagate( inputs[0], exp_outputs[0], outputs );

    // Sum further inputs
    for (int i = 1; i < inputs.size(); i++) {
        gradient_set addend = backpropagate( inputs[i], exp_outputs[i], outputs );
        for (int j = 0; j < addend.kernels_gradient.size(); j++) gradients.kernels_gradient[j] += addend.kernels_gradient[j];
        for (int j = 0; j < addend.cbiases_gradient.size(); j++) gradients.cbiases_gradient[j] += addend.cbiases_gradient[j];
        for (int j = 0; j < addend.weights_gradient.size(); j++) gradients.weights_gradient[j] += addend.weights_gradient[j];
        for (int j = 0; j < addend.biases_gradient.size(); j++) gradients.biases_gradient[j] += addend.biases_gradient[j];
    }
    // Divide by input size (i.e finish the average over all the sums)
    for (auto &k : gradients.kernels_gradient) k /= inputs.size();
    for (auto &cb : gradients.cbiases_gradient) cb /= inputs.size();
    for (auto &w : gradients.weights_gradient) w /= inputs.size();
    for (auto &b : gradients.biases_gradient) b /= inputs.size();

    // NN backprop (confirmed works)
    for (int i = 0; i < layers; i++) {
        weights[i] -= learning_rate * gradients.weights_gradient[i];
        biases[i] -= learning_rate * gradients.biases_gradient[i];
    }

}