import jetson.inference
import jetson.utils
import cv2
import numpy as np

# Initialize the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load the input image
input_image = "people1.jpeg"  # Your image
img = jetson.utils.loadImage(input_image)

# Detect objects in the image
detections = net.Detect(img)

# Convert CUDA image to OpenCV format for saving
cuda_mem = jetson.utils.cudaToNumpy(img)
cv_img = cv2.cvtColor(cuda_mem, cv2.COLOR_RGBA2BGR)

# Save the detection image
output_image = "detection_result.jpg"
cv2.imwrite(output_image, cv_img)

# Save detection results to text file
with open("detection_results.txt", "w") as file:
    file.write(f"Detection Results for {input_image}\n")
    file.write(f"Total objects detected: {len(detections)}\n")
    file.write("-" * 50 + "\n\n")
    
    for i, detection in enumerate(detections, 1):
        result = f"""Detection #{i}:
Class ID: {detection.ClassID}
Class Name: {net.GetClassDesc(detection.ClassID)}
Confidence: {detection.Confidence:.2f}
Bounding Box:
    Left:   {detection.Left:.2f}
    Top:    {detection.Top:.2f}
    Right:  {detection.Right:.2f}
    Bottom: {detection.Bottom:.2f}
Center: X: {detection.Center[0]:.2f}, Y: {detection.Center[1]:.2f}
Area: {detection.Area:.2f}
"""
        file.write(result + "\n")
        print(result)  # Also print to console

# Display the image with detections
display = jetson.utils.videoOutput("display://0")
display.Render(img)

# Wait for user to close the window
try:
    while display.IsStreaming():
        display.Render(img)
        display.SetStatus(f"Objects Detected: {len(detections)} | Network FPS: {net.GetNetworkFPS():.0f}")
except KeyboardInterrupt:
    print("Program ended by user")

print(f"Detection image saved as '{output_image}'")
print(f"Detection results saved as 'detection_results.txt'")
