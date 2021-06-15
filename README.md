# RelativePoseCam


The goal of this task is to provide space displacement vector and euler displacement 
vector (difference of camera directions) for given pair of two neighbour frames in video.

This task is neural based, so we have to collect the data, and we will use colmap 
in console mode for this purpose. All actions are performed in nvidia-docker container, 
which will install colmap and useful apps. 


Then you can find sparse_model.sh bash script in storage/ folder, this script will prepare and preprocess
the data. Then in code/model path you can run train_eval.py script and provide paths with colmap sparse models to the
dataset class.


Solution. For this task I chase resnet-34 architecture for extract features for each image in current pair. 
Then I concatenate two feature tensors and provide it as input for 3-layer fully convolutional NN, 
consists of convolutional layers. The final layer produces 6-dimanesional vector (first 3 components - 
for angles, activation function - tanh, second 3 components - for displacement). The loss function is MAE, 
because I used videos from smartphone, and quality of frames in sample may be relatively bad. 
If I use MSE, the NN will adjust to relatively big number of outliers.
