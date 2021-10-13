# Dog-Breed Classifier Project
Objective: Write an Algorithm for a Dog Identification App
The code accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

### Steps
We break the notebook into separate steps.

- Step 0: Import Datasets
  The dataset contains total of 13233 human images and 8351 dog images classified in to 133 dog breeds.
 
- Step 1: Detect Humans                                                           Accuracy-96%
 In this section, we have used OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in the images.
 OpenCV provides many pre-trained face detectors, stored as XML files on github.
 Sample Image Output
![image](https://user-images.githubusercontent.com/73768660/137092412-65042a26-57f8-4939-8320-9538684d5344.png)


- Step 2: Detect Dog                                                              Accuracy-86%
 Used VGG 16 pretrained model for detecting Dogs and Human faces for Step 1 and 2.
 ![image](https://user-images.githubusercontent.com/73768660/136780939-200769d9-6116-49bb-885f-664257ef1270.png)
 I also tried pre-trained ResNet-50 and Inception_V3 models for classification. 


- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)                      Accuracy-15%
  Data Preprocessing: The training images are resized to 255 x 255 size and cropped at center to 224 x 224 size. The training data is augmented with random horizontal and      vertical flips as well as rotation. Data is further normalized before being fed to convolutional layers. 
  
  

- Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)           Accuracy-68% 
 Here I have used pretrained VGG-16 with modifications in fully connected linear layer 2 and 3 to provide 133 classification output (instead of 1000 classes) corresponding to     each dog breed. The convolutional layer features have been kept the same as for the pretrained VGG-16 model as the dog-breed dataset is similar to ImageNet dataset on which     VGG is pretrained and Dog dataset are similar. 

 VGG(
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=1000, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=1000, out_features=133, bias=True)
  )
 )

- Step 5: Testing Algorithm

Final Predictions

![image](https://user-images.githubusercontent.com/73768660/136780218-eb521b3e-ec7e-4bec-bfe1-657effc64c61.png)
  This dog breed is Bullmastiff

Hello Human
![image](https://user-images.githubusercontent.com/73768660/136780329-1e0fbdde-2b18-45af-a45f-9a6dd33eb3f1.png)
  You Look Like a Pharaoh hound



## Future tasks to make my project stand out

### 1 TURN YOUR ALGORITHM INTO A WEB APP
Turning the code into a web app using Flask.

### 2 ADD FUNCTIONALITY FOR DOG MUTTS
Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned. The algorithm is currently guaranteed to fail for every mixed breed dog. Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, need to find a nice balance.

### 3 EXPERIMENT WITH MULTIPLE DOG/HUMAN DETECTORS
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
