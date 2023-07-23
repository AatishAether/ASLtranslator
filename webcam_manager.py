import cv2
import numpy as np
import mediapipe as mp


WHITE_COLOR = (245, 242, 226)
RED_COLOR = (25, 35, 240)
BGCOLOR = (245, 242, 176, 0.85)
FONTCOLOR = (118, 62, 37)
FONT = cv2.FONT_HERSHEY_COMPLEX

HEIGHT = 720


class WebcamManager(object):
    """Object that displays the Webcam output, draws the landmarks detected and
    outputs the sign prediction
    """

    def __init__(self,showImg,showHands,showBody):
        self.sign_detected = ""
        self.show_hands = showHands
        self.show_body = showBody
        self.show_img = showImg

    def update(
        self, frame: np.ndarray, results, sign_detected: str, is_recording: bool, show_image: bool, show_hands: bool, show_body: bool
    ):
        handsFrame = np.zeros((480,640,3), dtype=np.uint8)
        self.sign_detected = sign_detected
        self.show_img = show_image
        self.show_body = show_body
        self.show_hands = show_hands
        # Draw landmarks
        self.draw_landmarks(frame, results)
        #self.draw_hands(handsFrame,results)

        WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
        # Resize frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # Flip the image vertically for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Write result if there is
        frame = self.drawResults(frame,self.sign_detected)
        frame = self.drawRecordButton(frame,is_recording) 
        frame = self.drawToggles(frame,self.show_img,self.show_body,self.show_hands)        
        # Update the frame
        cv2.imshow("OpenCV Feed", frame)
        #cv2.imshow("Skeleton",handsFrame)
    def drawRecordButton(self,frame,is_recording):
        #Choose color
        circle_color = RED_COLOR if is_recording else WHITE_COLOR
        cv2.circle(frame, (30, 30), 20, circle_color, -1)
        return frame
        
    def drawResults(self,frame,sign_detected):
        font_size = 1
        font_thickness=2
        offset = int(HEIGHT * 0.02)
        window_w = int(HEIGHT  * len(frame[0]) / len(frame))

        (text_w, text_h), _ = cv2.getTextSize(sign_detected, FONT, font_size, font_thickness)
        text_x, text_y = int((window_w - text_w)/2), HEIGHT - text_h - offset

        cv2.rectangle(frame, (0, text_y - offset), (window_w, HEIGHT), BGCOLOR, -1)
        cv2.putText(
            frame,
            sign_detected,
            (text_x, text_y + text_h + font_size - 1),
            FONT,
            font_size,
            FONTCOLOR,
            font_thickness,
        )
        return frame
    
    def drawToggles(self,frame,show_img,show_hands,show_body):
        font_size = 1
        font_thickness=1
        offset = int(HEIGHT * 0.02)
        window_w = int(HEIGHT  * len(frame[0]) / len(frame))

        ###HANDS
        text = "Hands"
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_size, font_thickness)
        text_x,text_y = int(window_w - text_w)-offset,text_h+offset
        cv2.putText(frame,
                    text,(text_x,text_y+text_h+font_size - 1),
                    FONT,font_size,FONTCOLOR,font_thickness)
        cv2.createButton("Hands",self.draw_landmarks,None,cv2.QT_PUSH_BUTTON,1)

        ###BODY
        text = "Body"
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_size, font_thickness)
        text_x,text_y = int(window_w - text_w)-offset,text_h+offset
        cv2.putText(frame,
                    text,(text_x,text_y+text_h*2+font_size*2 - 1),
                    FONT,font_size,FONTCOLOR,font_thickness)

        ###IMAGE
        text = "Image"
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_size, font_thickness)
        text_x,text_y = int(window_w - text_w)-offset,text_h+offset
        cv2.putText(frame,
                    text,(text_x,text_y+text_h*3+font_size*3 - 1),
                    FONT,font_size,FONTCOLOR,font_thickness)
        return frame 

    @staticmethod
    def draw_landmarks(image, results):
        mp_holistic = mp.solutions.holistic  # Holistic model
        mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
        mp_drawing_styles = mp.solutions.drawing_styles

        # Draw left hand connections
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec = mp_drawing_styles.get_default_hand_connections_style()
            ) 
        # Draw right hand connections
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec = mp_drawing_styles.get_default_hand_connections_style()
            )

        ##Black frame
        # Draw left hand connections
        #mp_drawing.draw_landmarks(
        #    handsFrame,
        #    landmark_list=results.left_hand_landmarks,
        #    connections=mp_holistic.HAND_CONNECTIONS,
        #    landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style(),
        #    connection_drawing_spec = mp_drawing_styles.get_default_hand_connections_style()
        #    ) 
        ## Draw right hand connections
        #mp_drawing.draw_landmarks(
        #    handsFrame,
        #    landmark_list=results.right_hand_landmarks,
        #    connections=mp_holistic.HAND_CONNECTIONS,
        #    landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style(),
        #    connection_drawing_spec = mp_drawing_styles.get_default_hand_connections_style()
        #    )
        #mp_drawing.draw_landmarks(
        #    image,
        #    landmark_list=results.pose_landmarks,
        #    connections=mp_holistic.POSE_CONNECTIONS,
        #    landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style(),
        #)
        #mp_drawing.draw_landmarks(
        #    handsFrame,
        #    results.face_landmarks,
        #    mp_holistic.FACEMESH_CONTOURS,
        #    landmark_drawing_spec=None,
        #    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        #)
        
       
   
