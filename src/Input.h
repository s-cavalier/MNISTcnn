#ifndef M_LOADER
#define M_LOADER
#include <utility>
#include <vector>
#include <string>

#include "../Dependencies/eigen/Eigen/Dense"

struct MNIST_loader {
    // Shorthand for pair<vector<vector<uchar>>, uchar>
    // first = Image, second = Label
    using image_pair = std::pair<std::vector<std::vector<unsigned char>>, char>;

    //Loads an image and label set. Can be used for training or testing set.
    MNIST_loader(const std::string &image_set_path, const std::string &label_set_path, const unsigned int &set_size = 60000);

    //Returns a const reference to an image pair. Should not be used to modify set. Is exception safe.
    const image_pair& operator [](const unsigned int &index) const;

    void add_noised_images(const int &start, const int &end, const Eigen::Matrix2f &transformation);

    //Returns image set size.
    int size() const;

    //Returns a const iterator to the image set.
    std::vector<image_pair>::const_iterator begin() const;

    //Returns a const iterator to the end of the image set.
    std::vector<image_pair>::const_iterator end() const;

    //Very crude output tester.
    void image_tester(const image_pair &pair) const;

private:
    //Flips bytes in place.
    void flip_bytes(int &input);

    //Vector of image pairs. Not to be modified.
    std::vector<image_pair> labelled_image_set;
};

#endif