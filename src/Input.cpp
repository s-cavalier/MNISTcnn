#include "Input.h"
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

void MNIST_loader::flip_bytes(int &input) {
    input = _byteswap_ulong(input);
}

int MNIST_loader::size() const {
    return labelled_image_set.size();
}

const MNIST_loader::image_pair& MNIST_loader::operator [](const unsigned int &index) const {
    return labelled_image_set[index];
}

std::vector<MNIST_loader::image_pair>::const_iterator MNIST_loader::begin() const {
    return labelled_image_set.begin();
}

std::vector<MNIST_loader::image_pair>::const_iterator MNIST_loader::end() const {
    return labelled_image_set.end();
}

MNIST_loader::MNIST_loader(const std::string& image_set_path, const std::string& label_set_path, const unsigned int &set_size) {

    using uchar = unsigned char;

    //Open file
    std::ifstream images(image_set_path, std::ios::in | std::ios::binary);
    std::ifstream labels(label_set_path, std::ios::in | std::ios::binary);

    //Check file is opened properly
    if (!images.is_open()) throw std::runtime_error("Failed to open images file at: " + image_set_path);
    if (!labels.is_open()) throw std::runtime_error("Failed to open labels file at: " + label_set_path);

    //Check magic numbers
    int images_mn;
    int labels_mn;
    images.read((char*)&images_mn, 4);
    labels.read((char*)&labels_mn, 4);

    flip_bytes(images_mn);
    flip_bytes(labels_mn);

    if (images_mn != 2051) throw std::runtime_error("Images magic number failed to match 2051. Recieved " + images_mn);
    if (labels_mn != 2049) throw std::runtime_error("Images magic number failed to match 2051. Recieved " + labels_mn);

    //File has been read correctly

    //Image headers
    int images_count, img_rows, img_cols;
    images.read((char*)&images_count, 4);
    images.read((char*)&img_rows, 4);
    images.read((char*)&img_cols, 4);
    flip_bytes(images_count);
    flip_bytes(img_rows);
    flip_bytes(img_cols);

    //Label headers
    int labels_count;
    labels.read((char*)&labels_count, 4);
    flip_bytes(labels_count);

    //Assert that images & label count match for correct label matching.
    if (images_count != labels_count) throw std::runtime_error("Image count and label count mismatch. Image count & label count:" + std::to_string(images_count) + "," + std::to_string(labels_count));
    if (set_size > images_count) throw std::runtime_error("Image count and label count mismatch. Image count & preset size (based on parameters):" + std::to_string(images_count) + "," + std::to_string(set_size));

    images_count = set_size;

    //Read all images & labels.
    for (int i = 0; i < images_count; i++) {
        image_pair pair;
        pair.second = 0;

        //Set label first.
        labels.read(&pair.second, 1);

        //Set image.
        for (int i = 0; i < img_rows; i++) {
            pair.first.push_back({std::vector<unsigned char>()});
            for (int j = 0; j < img_cols; j++) {
                pair.first[i].push_back(uchar());
                images.read((char*)&pair.first[i][j], 1);
            }
        }

        labelled_image_set.push_back(pair);

    }

}

void MNIST_loader::image_tester(const image_pair &pair) const {
    std::cout << (int)pair.second << std::endl;
    std::stringstream ss;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            ss << '(' << std::setfill('0') << std::setw(3) << (int)pair.first[i][j] << ") ";
        }
        ss << std::endl;
    }
    
    std::string out;
    while (std::getline(ss, out)) {
        std::cout << out << std::endl;
    }
}

