#!/usr/bin/env python3

"""
Sample demo application for face recognition.

Loads face images from the local drive
The images are already preprocess to be compatible with Resnet50
For using it with sphereface the images would have to be rescaled to 112x112.
The tile is then sent to a remote HTTP resnet50 or sphereface worker to perform inference.
The result is then collected and the cosine similarity between the first image is computed an printed as a list.

"""

import sys
import os
import argparse
from sklearn.metrics import pairwise_distances
from typing import List
from abc import abstractmethod, ABCMeta
import numpy
import requests
import cv2
import matplotlib.pyplot as plt


# Add our project directory
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
class Worker(metaclass=ABCMeta):
    """ Process data """
    @abstractmethod
    def process(self, input_data: numpy.array = None) -> numpy.array:
        """
        Process data.
        :returns: precessed results
        """
        pass


class WorkerProxyHttp(Worker):
    """
    Process data using a worker connected via HTTP
    The remote worker will receive and produce results as float arrays
    """
    def __init__(self, ip_addr: str, port : int):
        self._ip_addr = ip_addr
        self._port = port

    def process(self, input_data: numpy.array = None) -> numpy.array:
        """
        Send a POST request with the input data to the HTTP server.
        Extract processed results from the reply content.
        :returns: precessed results
        """
        url = 'http://' + self._ip_addr + ':' + str(self._port) + '/artifacts/'
        with requests.post(url, data=input_data.astype(numpy.float32).tobytes()) as req:
            if req.status_code != 200:
                raise Exception("Error returned by the server worker")
            results = numpy.frombuffer(req.content, dtype=numpy.float32).reshape(1, -1)
        return results


def print_bold(text):
    print("\033[1m" + text + "\033[0m")

def load_images():
    images = []
    img = cv2.imread('face1.jpeg')
    images.append(img)
    img = cv2.imread('face2.jpg')
    images.append(img)
    img = cv2.imread('face3.jpg')
    images.append(img)
    return images
def main():
    parser = argparse.ArgumentParser(description="Face recognition demo application")
    parser.add_argument('--ip_addr', required=True, help="IP address of HTTP inference worker")
    parser.add_argument('--ip_port', required=True, help="IP port number of HTTP inference worker")
    args = parser.parse_args()
    image_features = []
    
    # Create the app
    print_bold("Processing...")
    try:
        # Create the tools and workers we need
        inference_processor = WorkerProxyHttp(args.ip_addr, int(args.ip_port))

        # Process input images
        for img in load_images():
            print_bold('processing image')            
            inference_result = inference_processor.process(img)
            inference_result = inference_result.flatten()
            image_features.append(inference_result)
        results = pairwise_distances(image_features, metric='cosine')
        print(results[0])
    finally:
        # Tear down
        print_bold("Done!")


if __name__ == "__main__":
    main()
