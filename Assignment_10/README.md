## This assignment is about Session 10 - Residual Connections in CNNs and One Cycle Policy

# Our Assignment was to
1. Write a Architecture like
	1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    
	2. Layer1 -
		1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
		2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
		3. Add(X, R1)
     
	3. Layer 2 -
		1. Conv 3x3 [256k]
		2. MaxPooling2D
		3. BN
		4. ReLU
     
	4. Layer 3 
		1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
		2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
		3. Add(X, R2)
     
	5. MaxPooling with Kernel Size 4
    
	6. FC Layer
    
	7. SoftMax
2. Uses One Cycle Policy such that:
	1. Total Epochs = 24
	2. Max at Epoch = 5
	3. LRMIN = FIND
	4. LRMAX = FIND
	5. NO Annihilation

3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Use ADAM and CrossEntropyLoss
6. Target Accuracy: 90%
7. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. 
	I should be able to find the custom_resnet.py model in your MASTER GitHub repo that you'd be training. repo that you'd be training. yLoss
Target Accuracy: 90%


#### Inside models folder there is Assignment_10_models.py file is there, that file contains the arhitecture of our models
#### Inside One_Cycle_Policy folder there isOne_Cycle_Policy.ipynb folder is there which contains the LR Finder code to find LRMIN and LRMAX.
#### Inside Miscellaneous folder there is utils.py file which contains Albumentation Transformation code, train loader and test loader codes,etc

# Results
1. Parameters: 6,614,080

2. Best Train Accuracy: 88.09% at 24th epoch
   
3. Best Test Accuracy: 88.05 at 24th epochh)

# Analysis
1. We have implemented our model like ResNet architecture

2. We implemented Adam optimizer and CrossEntropyLoss and got higher and faster accuracy according to our model architecture and problem statment

3. Used albumentation transformation library of RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)

4. If we train our model for higher epoch then we can get slighlty more accuracy


In Assignment_10.ipynb file contains our actual models traning logs. As mentioned in assignment that ipython notebook should import package from our Github folder. So this file clone the repository and directly imported the packages from our github 

