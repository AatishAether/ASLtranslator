import cv2
import mediapipe as mp
import os
import sys
import pickle
import pandas as pd

import re

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#mp_hands = mp.solutions.hands

##Init Vars
video = str(sys.argv[1])
if sys.argv[1] == 'stream':
    video = 0

showImg = True
showHands = True
showBody = True


def main():
    # Create dataset of the videos where landmarks have not been extracted yet
    print("Reading Dataset...")
    #n is num of new vids
    n,dataset = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int) 
    print("Loading Signs...")
    if not (os.path.exists("./referenceSigns.pickle")) or (n > 0):
        reference_signs = load_reference_signs(dataset)

        reference_signs.to_pickle('./referenceSigns.pickle')

    else:
        reference_signs = pd.read_pickle('./referenceSigns.pickle')


    
    print("Creating Sign Recorder object")
    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager(showImg,showHands,showBody)

    # Turn on the webcam
    cap = cv2.VideoCapture(video, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (Stream end?)...")


            # Make detections of hands or body (or face)
            image, mpResults = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(mpResults)


            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, mpResults, sign_detected, is_recording, showImg, showHands, showBody)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break
            elif pressedKey == ord('p'): ##Print to file
                features = sign_recorder.recorded_sign.lh_embedding
                #features = str(features).replace('[','')
                #features = features.replace(']','')
                #features = features.replace("'",'')
                #features = list(features.split(","))
                #features = list(map(float,features))
                #print(features)
                with open("openHand.pickle", "wb") as f:
                    pickle.dump(features,f)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
