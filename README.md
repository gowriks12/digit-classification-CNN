# digit-classification-CNN
Digit classification using neural networks is implemented in this project.

Dataset: Download the MNIST data (60,000 training images and 10,000 testing images) from the 
data source, http://yann.lecun.com/exdb/mnist/
The MNIST dataset include the 10 digit labels, from 0 to 9. Each (grayscale) image is a 28 x 28 matrix of pixels, with values between 0 and 255. Each pixel is converted to a value in the interval [0, 1] by dividing by 255. The figure below shows an example of each digit from the MINST dataset. 
Image data structure: Since images are 2-dimensional matrices, we first flatten them into a vector 𝐱 ∈ ℝ784 with dimensionality 𝑑 = 28 × 28 = 784. This is done by simply concatenating all of the 
rows of the images to obtain one long vector.
Output labels: Since the output labels are categorical values that denote the digits from 0 to 9, we 
need to convert them into binary (numerical) vectors, using one-hot encoding. Thus, the label 0 is 
encoded as 𝒆1 = (1,0,0,0,0,0,0,0,0,0) 𝑇 ∈ ℝ10, the label 1 as 𝒆2 = (0,1,0,0,0,0,0,0,0,0)

𝑇 ∈ ℝ10, and so on, and finally the label 9 is encoded as 𝒆10 = (0,0,0,0,0,0,0,0,0,1)
𝑇 ∈ ℝ10.
Input and output: Each input image vector 𝐱 has a corresponding target response vector𝐲 ∈
{𝒆1, 𝒆2, … , 𝒆10}. 
Neural network: We will consider three different neural network models: (i) a neural network with no hidden layer, (ii) a neural network with one hidden layer with 7 neurons, and (iii) a neural network with one hidden layer with 49 neurons. In all the models, the input layer has 𝑑 = 784 and the output layer has 10 neurons.
Training: We train each neural network for 10 epochs, using learning rate (/step size) 𝛼 = 0.25. 
Evaluation: During training each neural network model (i), (ii), (iii), count the number of misclassified images after each epoch, on the separate MNIST test (/validation) set comprising 10,000 images. At the end of training (i.e., after 10 epochs), report the final number of test (/validation) errors. After building all the three neural network models, plot is generated to show the number of errors (y axis) per each epoch (x axis) of the three models. 

![image](https://user-images.githubusercontent.com/82420256/146601085-70307c9b-e8d1-4000-83d7-b031ca8bdda6.png)

