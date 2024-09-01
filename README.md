# My MNIST Convolutional Neural Network
Serves mainly as a personal proof-of-concept for convolutions. <br>
Has a built class for performing 1D and 2D personally made fast fourier convolutions. <br>
CMatrix class still has a lot of room open for optimizations, however it is functional and learns to around 70% in one epoch. <br>

The key of this repo, however, is that it contains a special algorithim for max-pooling that operations in `O(1)` constant space (besides making the output matrix itself, although an in-place operation is very possible) and `O(n + m)` linear time. It scales particularly well when the size of a max-pool kernel doesn't equal its stride. Please do not use without crediting.
