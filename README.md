#Dog-Breed Classifier Project
Objective: Write an Algorithm for a Dog Identification App
The code accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).

### Steps
We break the notebook into separate steps.

Step 0: Import Datasets
The dataset consists of separate libraries of dog faces and humand faces. 

Step 1: Detect Humans 
In this step we use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. 

Step 2: Detect Dogs
Here we use a pre-trained model of VGG-16 to detect dogs. The VGG-16 is obtained from torchvision.models

Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)

Step 5: Write your Algorithm

Step 6: Test Your Algorithm
