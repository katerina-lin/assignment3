#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

# Initialize the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load the input image
input_image = "people1.jpeg"  # Replace with your image path
img = jetson.utils.loadImage(input_image)

# Detect objects in the image
detections = net.Detect(img)

# Create output text file
with open("detection_results.txt", "w") as file:
    file.write(f"Total objects detected: {len(detections)}\n")
    file.write("-" * 50 + "\n")
    
    # Print and save detection results
    for i, detection in enumerate(detections, 1):
        # Format the detection information
        result = f"""
Detection #{i}:
Class ID: {detection.ClassID}
Class Name: {net.GetClassDesc(detection.ClassID)}
Confidence: {detection.Confidence:.2f}
Bounding Box:
    Left:   {detection.Left:.2f}
    Top:    {detection.Top:.2f}
    Right:  {detection.Right:.2f}
    Bottom: {detection.Bottom:.2f}
Center:
    X: {detection.Center[0]:.2f}
    Y: {detection.Center[1]:.2f}
Area: {detection.Area:.2f}
"""
        # Print to console
        print(result)
        # Write to file
        file.write(result + "\n")

# Display the image with detections
display = jetson.utils.videoOutput("display://0")
display.Render(img)

# Wait for user to close the window
try:
    while display.IsStreaming():
        display.Render(img)
        # Add detection info as status
        display.SetStatus(f"Objects Detected: {len(detections)} | Network FPS: {net.GetNetworkFPS():.0f}")
except KeyboardInterrupt:
    print("Program ended by user")

print(f"Results have been saved to 'detection_results.txt'")
