// STD Imports
#include <iostream>
#include <iomanip>

// Local Dependencies
#include "Input.h"
#include "cNetwork.h"

// Imports
#include "../Dependencies/eigen/Eigen/Dense"

using std::cout, std::endl;

Eigen::MatrixXd stdvec28x28_to_eigenmat28x28(const std::vector<std::vector<unsigned char>> &vec) {
    Eigen::MatrixXd m(28, 28);
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            m(r, c) = ( ((double)vec[r][c]) / 255);
        }
    }
    return m;
}


int main(int argc, char** argv) {

    std::cout << "Generating training data." << std::endl;
    MNIST_loader training_set("Data/MNIST/train-images-idx3-ubyte", "Data/MNIST/train-labels-idx1-ubyte", 60000);
    std::cout << "Generating testing data." << std::endl;
    MNIST_loader testing_set("Data/MNIST/t10k-images-idx3-ubyte", "Data/MNIST/t10k-labels-idx1-ubyte", 10000);
    
    std::cout << "Done. Generating network..." << std::endl;
    cNetwork network(28, { 1 }, { {3, 5}, {2, 2} }, { 30 }, 10);
    int batch_size = 10;
    network.learning_rate = 0.1;

    std::cout << network.activate( stdvec28x28_to_eigenmat28x28(training_set[0].first) ) << std::endl;


    for (int i = 0; i < training_set.size(); i += batch_size) {
        cout << "Training: Batch " << i / batch_size  << " [" << i << ", " << i + batch_size - 1 << "] / " << training_set.size() << endl;

        std::vector<Eigen::VectorXd> outputs;
        std::vector<Eigen::MatrixXd> img(batch_size);

        //Grab images
        for (int j = 0; j < img.size(); j++) img[j] = ( stdvec28x28_to_eigenmat28x28( training_set[i + j].first ) );

        // Generate corresponding "expected" output vectors
        std::vector<Eigen::VectorXd> expected(batch_size);
        for (int j = 0; j < expected.size(); j++) {
            expected[j].resize(10);
            for (int k = 0; k < expected[j].rows(); k++) expected[j](k) = (k == (int)training_set[i + j].second);
        }

        // Run backpropagation, save outputs
        network.train( img, expected, &outputs);
        for (int j = 0; j < outputs.size(); j++) {
            int label = (int)training_set[i + j].second;
            int max_ind = 0;
            double max = 0;
            for (int k = 0; k < outputs[j].size(); k++) {
                if (outputs[j](k) <= max) continue;
                max_ind = k;
                max = outputs[j](k);
            }

            cout << "Case " << i + j << ", Expected: " << label << ", Received: " << max_ind << ", Correct? " << std::boolalpha << (label == max_ind) << endl;  
        }
        cout << endl;

    }

    int correct = 0;

    // Test system for each image in testing set
    for (int i = 0; i < testing_set.size(); i++) {
         std::cout << "Testing: Processing image " << i + 1 << " / " << testing_set.size() << std::endl; 

        // Run and show results.
        Eigen::VectorXd output = network.activate( stdvec28x28_to_eigenmat28x28 (training_set[i].first) );

        float max = 0;
        int max_ind = 0;
        for (int j = 0; j < 10; j++) {
            if (max >= output(j)) continue;
            max = output(j);
            max_ind = j;
        }

        bool check = (max_ind == (int)training_set[i].second);
        correct += check;

        std::cout << "Expected: " << (int)training_set[i].second << ", Recieved: " << max_ind << ", Success? " << std::boolalpha << check << ", Accuracy: " << 100.0 * ((float)correct) / (i + 1) << "%" << std::endl << std::endl;

    }
    
    std::cout << "Finished with no errors, accuracy rate: " << 100.0 * ((float)correct) / testing_set.size() << "%" << std::endl;
    
    
    
    
}