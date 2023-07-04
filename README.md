#American Sign Language (ASL) Recognition w/ MediaPipe's Hollistic Model and Dynamic Time Warping (DTW)

![License: MIT](https://img.shields.io/badge/license-MIT-green)

This repository proposes an implementation of a Sign Recognition Model using the **MediaPipe** library 
for landmark extraction and **Dynamic Time Warping** (DTW) as a similarity metric between signs.
This is a fork of gabguerin's Sign-Language-Recognition--MediaPipe-DTW found here:
https://github.com/gabguerin/Sign-Language-Recognition--MediaPipe-DTW

*NOTE:* This utilizes the legacy MediaPipe solutions and is intended to upgrade to the newer models soon. Until then, strange behavior was experienced with versions of mediapipe > version 0.8.9.1. Python 3.7 was found to fit this version requirement. It is highly recommended to use a virtual environment and pip3.7 install requirements.txt

![](example.gif)

#### Source : https://www.sicara.ai/blog/sign-language-recognition-using-mediapipe
___

## Set up

### 1. Open terminal and go to the Project directory

### 2. Install the necessary libraries

- ` pip3.7 install -r requirements.txt `

### 3. Import Videos of signs which will be considered as reference
The architecture of the `videos/` folder must be:
```
|data/
    |-videos/
          |-Hello/
            |-<video_of_hello_1>.mp4
            |-<video_of_hello_2>.mp4
            ...
          |-Thanks/
            |-<video_of_thanks_1>.mp4
            |-<video_of_thanks_2>.mp4
            ...
```


### 4. Load the dataset and turn on the Webcam

- ` python main.py `

### 5. Press the "r" key to record the sign. 

___
## Code Description

### *Landmark extraction (MediaPipe)*

- The **Holistic Model** of MediaPipe allows us to extract the keypoints of the Hands, Pose and Face models.
For now, the implementation only uses the Hand model to predict the sign.


### *Hand Model*

- In this project a **HandModel** has been created to define the Hand gesture at each frame. 
If a hand is not present we set all the positions to zero.

- In order to be **invariant to orientation and scale**, the **feature vector** of the
HandModel is a **list of the angles** between all the connections of the hand.

-Per frame basis

### *Sign Model*

- The **SignModel** is created from a list of landmarks (extracted from a video)

- For each frame, we **store** the **feature vectors** of each hand.

-Per Time Series basis

### *Sign Recorder*

- The **SignRecorder** class **stores** the HandModels of left hand and right hand for each frame **when recording**.
- Once the recording is finished, it **computes the DTW** of the recorded sign and 
all the reference signs present in the dataset.
- Finally, a voting logic is added to output a result only if the prediction **confidence** is **higher than a threshold**.

### *Dynamic Time Warping*

-  DTW is widely used for computing time series similarity.

- In this project, we compute the DTW of the variation of hand connection angles over time.

___

## *Current Limitations*

- Needs Pose Feature Vectorization, then a Hollistic model that compares Hand features to Pose features.

- Needs a similar approach for Face features. Considering a Semantic/Emotional Analysis model built in to the Face model

- Once Pose Feature Vectorization is realized, larger annotated datasets can be brought in

- Once larger datasets are brought in, DTWnet can be implemented

## References

 - [Pham Chinh Huu, Le Quoc Khanh, Le Thanh Ha : Human Action Recognition Using Dynamic Time Warping and Voting Algorithm](https://www.researchgate.net/publication/290440452)
 - [Mediapipe : Pose classification](https://google.github.io/mediapipe/solutions/pose_classification.html)
