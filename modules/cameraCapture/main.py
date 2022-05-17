# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import time
import sys
import os
import requests
import json
import cv2
from azure.iot.device import IoTHubModuleClient, Message
from flask import Flask, render_template, Response
from imutils.video import FileVideoStream, VideoStream
import imutils 

app = Flask(__name__)

# global counters
SENT_IMAGES = 0

# global client
CLIENT = None

# Send a message to IoT Hub
# Route output1 to $upstream in deployment.template.json
def send_to_hub(strMessage):
    message = Message(bytearray(strMessage, 'utf8'))
    CLIENT.send_message_to_output(message, "output1")
    global SENT_IMAGES
    SENT_IMAGES += 1
    print( "Total images sent: {}".format(SENT_IMAGES) )

# Send an image to the image classifying server
# Return the JSON response from the server with the prediction result
def sendFrameForProcessing(imagePath, imageProcessingEndpoint):
    headers = {'Content-Type': 'application/octet-stream'}

    #with open(imagePath, mode="rb") as test_image:

    #test_image = imagePath
    
    try:
        response = requests.post(imageProcessingEndpoint, headers = headers, data = imagePath)
        print("Response from classification service: (" + str(response.status_code) + ") " + json.dumps(response.json()) + "\n")
    except Exception as e:
        print(e)
        print("No response from classification service")
        return None

    return response.json()
    # return json.dumps(response.json())


# def main():
#     try:
#         global CLIENT
#         CLIENT = IoTHubModuleClient.create_from_edge_environment()
#     except Exception as iothub_error:
#         print ( "Unexpected error {} from IoTHub".format(iothub_error) )
#         return


def gen_frames(): 
    counter = 0 
    while True:
        frame = vs.read()
        # frame = cv2.resize(frame, (640, 480))

        faces = face_cascade.detectMultiScale(frame, 1.2, 4, minSize=(30,30))
        for (x,y,w,h) in faces:
            rec = cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 169, 201), 2)
            try:
                cv2.putText(rec, tag, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 100, 200), 5)
                cv2.putText(rec, prob, (x+400, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 100, 200), 5)
                cv2.putText(rec, "TagID:"+tagID, (x+300, y+700), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 100, 200), 5)
            except: None
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        if counter >= 60:
            classification = sendFrameForProcessing(frame, IMAGE_PROCESSING_ENDPOINT)
            for pred in classification['predictions']:
                if pred['probability'] > .25:
                    tag = pred['tagName']
                    prob = str(round(pred['probability'],2))
                    tagID = str(pred['tagId'])
            counter = 0
        elif counter == 0:
            classification = sendFrameForProcessing(frame, IMAGE_PROCESSING_ENDPOINT)
            for pred in classification['predictions']:
                if pred['probability'] > .25:
                    tag = pred['tagName']
                    prob = str(round(pred['probability'],2))
                    tagID = str(pred['tagId'])
        counter += 1
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        # Retrieve the image location and image classifying server endpoint from container environment
        VIDEO_PATH = os.getenv('IMAGE_PATH', "")
        IMAGE_PROCESSING_ENDPOINT = os.getenv('IMAGE_PROCESSING_ENDPOINT', "")
    except ValueError as error:
        print ( error )
        sys.exit(1)

    if ((VIDEO_PATH and IMAGE_PROCESSING_ENDPOINT) != ""):
        # vs = FileVideoStream(path = VIDEO_PATH).start()
        vs = VideoStream(src = VIDEO_PATH).start()
        # time.sleep(1.0)
        try:
            CLIENT = IoTHubModuleClient.create_from_edge_environment()
        except Exception as iothub_error:
            print ( "Unexpected error {} from IoTHub".format(iothub_error) )
        # cap = cv2.VideoCapture(VIDEO_PATH)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        # main()
        app.run(debug=True, port=8080, host='0.0.0.0')
        
    else: 
        print ( "Error: Image path or image-processing endpoint missing" )
