from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
from Adafruit_IO import MQTTClient  
import time

# Disable scientific notation for clarity
#np.set_printoptions(suppress=True)
AIO_USERNAME = "lnkduy"
dAIO_KEY = "aio_CAix72HVJYmbR7rn6muupuNaa3Wx"
client = MQTTClient(AIO_USERNAME , dAIO_KEY)
client.connect()
# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer

def person_detect():
        
    while True:
        camera = cv2.VideoCapture(0)
        # Grab the webcamera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")


        time.sleep(2)

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            camera.release()
            cv2.destroyAllWindows()
            return class_name[2:], int(str(np.round(confidence_score * 100))[:-2])
    


detect = person_detect()
client.publish("Confidence",  detect[1])
time.sleep(0.5)
client.publish("recognition", detect[0])