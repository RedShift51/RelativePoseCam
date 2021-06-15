# RelativePoseCam


The goal of this task is to provide space displacement vector and euler displacement 
vector (difference of camera directions) for given pair of two neighbour frames in video.

This task is neural based, so we have to collect the data, and we will use colmap 
in console mode for this purpose. All actions are performed in nvidia-docker container, 
which will install colmap and useful apps. 


Then you can find sparse_model.sh bash script in storage/ folder, this script will prepare and preprocess
the data. Then in code/model path you can run train_eval.py script and provide paths with colmap sparse models to the
dataset class.


Solution. For this task I used resnet-34 architecture for extract features for each image in current pair. 
Then I concatenate two feature tensors and provide it as input for 3-layer fully convolutional NN, 
consists of convolutional layers. The final layer produces 6-dimanesional vector (first 3 components - 
for angles, activation function - tanh, second 3 components - for displacement). The loss function is MAE, 
because I used videos from smartphone, and quality of frames in sample may be relatively bad. 
If I use MSE, the NN will adjust to relatively big number of outliers.

Example of inference and training / validation loss you can see below.

![0](https://user-images.githubusercontent.com/29106459/122049565-4004d800-cdeb-11eb-9023-99af634ef3a5.png)
![10](https://user-images.githubusercontent.com/29106459/122049581-44c98c00-cdeb-11eb-9128-1ed13e283c39.png)


angle difference (radians) [-0.2180,  0.2343,  0.2135]
displacement               [-0.1910, -0.1233,  0.2215]



Learning, losses
![mae_angle](https://user-images.githubusercontent.com/29106459/122051475-514ee400-cded-11eb-831a-36cbd92ff1b1.png)
![mae_displ](https://user-images.githubusercontent.com/29106459/122052173-126d5e00-cdee-11eb-9646-62c583d8d204.png)



