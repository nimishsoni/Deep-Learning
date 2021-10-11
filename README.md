#Dog-Breed Classifier Project
Objective: Write an Algorithm for a Dog Identification App
The code accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).

### Steps
We break the notebook into separate steps.
- Intro
- Step 0: Import Datasets
- Step 1: Detect Humans                                                           Accuracy-98%
- Step 2: Detect Dog                                                              Accuracy-98%
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)                      Accuracy-16%
- Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)           Accuracy-81% 
- Step 5: Writing Own Algorithm
- Step 6: Testing Own Algorithm

### Main CNN Model

![Model Architecture](model_architecture.PNG)

I used this architecture in the step 3 


and I used VGG-16 for the transfer learning in step 4. Here is the architecture of VGG-16:

![VGG16 Architecture](vgg16_architecture.png)
 I also tried pre-trained ResNet-50 and Inception_V3 models for classification. 

 ## Final Prediction
 
 
![Prediction](prediction.PNG)


## Future tasks to make my project stand out

### 1 AUGMENT THE TRAINING DATA
Augmenting the training and/or validation set might help improve model performance. 
### (DONE NOW)

### 2 TURN YOUR ALGORITHM INTO A WEB APP
Turning the code into a web app using Flask.

### 3 ADD FUNCTIONALITY FOR DOG MUTTS
Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned. The algorithm is currently guaranteed to fail for every mixed breed dog. Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, need to find a nice balance.

### 4 EXPERIMENT WITH MULTIPLE DOG/HUMAN DETECTORS
Perform a systematic evaluation of various methods for detecting humans and dogs in images & Provide improved methodology for the face_detector and dog_detector functions.




### Libraries

The list below represents main libraries and its objects for the project.
- [PyTorch](https://pytorch.org/) (Convolutional Neural Network)
- [OpenCV](https://opencv.org/) (Human Face Detection)
- [Matplotlib](https://matplotlib.org/) (Plot Images)
- [Numpy](http://www.numpy.org/) 

### Dataset
* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

#### Amazon Web Services

I Used Amazon Web Services to launch a GPU instance.
