# Dog-Breed Classifier Project
Objective: Write an Algorithm for a Dog Identification App
The code accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 

### Steps
We break the notebook into separate steps.
- Intro
- Step 0: Import Datasets
- Step 1: Detect Humans                                                           Accuracy-96%
- Step 2: Detect Dog                                                              Accuracy-86%
Used VGG 16 pretrained model for detecting Dogs and Human faces for Step 1 and 2.
![image](https://user-images.githubusercontent.com/73768660/136780939-200769d9-6116-49bb-885f-664257ef1270.png)
 I also tried pre-trained ResNet-50 and Inception_V3 models for classification. 


- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)                      Accuracy-15%

- Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)           Accuracy-68% 
Here I have used pretrained VGG-16 with modifications in fully connected linear layer 2 and 3 to provide 133 classification output corresponding to each dog breed. 
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
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
