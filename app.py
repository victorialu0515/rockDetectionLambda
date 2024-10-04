import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import numpy as np

from utils import decodeString

from sklearn.cluster import KMeans

import cv2
import json
import time

import os

import boto3
from tempfile import TemporaryDirectory

import base64

# import onnxruntime

from dataclasses import dataclass
from typing import List
from datetime import datetime

# import requests

@dataclass
class RockDetection:
  id: int
  bbox: List[float]
  colour: List[float]
  track: int

  def to_dict(self):
    return {"id":self.id, "bbox":self.bbox, "colour":self.colour, "track":self.track}

def getColour(image, result):
  colours = []
  imageArray = cv2.imread(image)
  for i in result:
    x, y, w, h = i

    img = imageArray[int(y-h/3):int(y+h/3), int(x-w/3):int(x+w/3)]
    average = img.mean(axis=0).mean(axis=0)

    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    colours.append(dominant[::-1])
  return colours

def clustering(numClusters, colours):
    kmeans = KMeans(n_clusters=numClusters, random_state=None)
    kmeans.fit(colours)
    labels = kmeans.labels_
    return list(labels)

def drawColour(image, result, output):
    imageArray = cv2.imread(image)
    for i, rectangle in enumerate(result):
        x, y, w, h = rectangle['bbox']
        r, g, b = rectangle['colour']
        cv2.rectangle(imageArray, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (int(b), int(g), int(r)), 5)
        cv2.putText(imageArray, str(rectangle['id']), (int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)
        cv2.putText(imageArray, str(rectangle['track']), (int(x), int(y-h/2-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(int(b), int(g), int(r)), thickness=5)
    cv2.imshow("image", imageArray)
    cv2.waitKey(0)

    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        tmp_image_path = 'rockDetection.jpg'
        cv2.imwrite(tmp_image_path, imageArray)
        with open(tmp_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        s3Key = f"rockdetection-{datetime.now().strftime('%Y, %m, %d, %H, %M, %S')}.jpg"
        upload_file(tmp_image_path, "rockdetectionbucket", s3Key)

        return encoded_string

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def detection(image_path, numCluster, output_path, model_path="best.pt"):
    model  = YOLO(model_path)
    results = model(image_path)
    result = [x.tolist() for x in results[0].boxes.xywh]
    resultArray = np.array(result)
    colours = getColour(image_path, result)
    tracks = clustering(numCluster, colours)
    list_colours = [array.tolist() for array in colours]
    list_rocks: List[RockDetection] = [
        RockDetection(id=i, bbox=bbox, colour=colour, track=int(track)).to_dict()
        for i, (bbox, colour, track) in enumerate(zip(result, list_colours, tracks))
    ]

    result_dict = {
        "user_id": "Victoria",
        "run_id": 0,
        "track_chosen": 1,
        "rocks": list_rocks,
    }
    # with open(f"{output_path}.json", "w") as f:
    #     json.dump(result_dict, f, indent=4)


    imageString = drawColour(image_path, result_dict["rocks"], output_path)
    return result_dict, imageString


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    body = event
    if event.get("httpMethod") == "POST":
        body = json.loads(event.get("body"))
    elif event.get("httpMethod") == "GET":
        body = event.get('queryStringParameters', event)
    image_b64 = body.get("image")
    model_path = body.get("model_path", "models/best.pt")

    # TODO: in lambda the image path needs to be tmp folder
    image_path = '/tmp/decode.jpg'
    decodeString(image_b64, image_path)
    results, imagestring = detection(image_path, 6, "test", model_path)


    return {
        "statusCode": 200,
        "body": json.dumps({
            "result_json": results,
            "image_string": imagestring
            # "location": ip.text.replace("\n", "")
        })
    }

if __name__ == "__main__":
    with open("/Users/victorialu/Downloads/IMG_2271_time_3.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    with open("imageString.txt", "w") as f:
        f.write(encoded_string)

    event = {
        "httpMethod": "GET",
        "queryStringParameters":
            {
                "image": encoded_string,
                "model_path": 'models/best.pt',
            },
    }

    t0 = time.time()
    lambda_handler(event, None)
    print(f"Time taken: {round(time.time() - t0, 3)} s")
