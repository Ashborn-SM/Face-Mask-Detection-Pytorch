## About
This is a course project i have created using pytorch.
Run the Detector_MTCNN.py. At present video is taken from the
webcam(live) you can change this by changing "cv.VideoCapture(0)".

**Make sure to download the state dict to get the predictions right**

**state dict- https://drive.google.com/drive/folders/1oRBDw_HmqCaQ2jnT4aSZHyYVBi4ELhSt?usp=sharing
dataset - https://drive.google.com/drive/folders/1LEKdePxk854r0kT542g42loM1z1UkL4g?usp=sharing**

I tried both the models with different video's, ResNet9 and ResNet15 performed well.
I noticed that there are some video ResNet9 performed well but ResNet15 did not and vice-versa.

Try both the models and see whats best.

## Third-Party Libraries used:
1. Facenet PyTorch
2. Open CV
3. PyTorch
4. Numpy
5. Matplotlib

## Guide
1. Guide to MTCNN in facenet-pytorch - https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
2. Facenet implementation in a video - https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb
