**FACE MASK DETECTION** 
---------------------------------------------------------------------
CAI D2 TEAM-28

**TEAM DETAILS**
--------------------------------------------------------------------
> SAI SANDEEP AOTINI(TL)

> SAI BHARGAV RUDRAPAKA

> MALLABATHULA SAI ATYA SURESH

> PARI MAHESH

> CHUKKA SIVA KARTHIK 

-------------------------------------------------------------------
Introduction

Face Mask Detection is a computer vision application that identifies whether individuals are wearing face masks in real-time. 
It has become crucial in ensuring public health and safety, especially during pandemics. 
This project employs  Deep Learning and OpenCV to develop an accurate and efficient face mask detection system.
Using a pre-trained deep learning model, the system can classify faces as "Mask" or "No Mask" with high accuracy.


Abstract

This project focuses on Face Mask Detection using Convolutional Neural Networks (CNNs) and OpenCV. 
The system captures real-time video feeds, processes the input using a trained model, and classifies faces into two categories: 
With Mask and Without Mask. 
This project has significant applications in enforcing mask-wearing policies in public areas and improving safety measures.


Technology

- Python: Core programming language used for development.
- OpenCV: Handles image and video processing tasks.
- TensorFlow/Keras: Provides deep learning models for face mask classification.
- CNN Model: A pre-trained or custom-built convolutional neural network is used for classification.


Uses and Applications

Face Mask Detection has a wide range of real-world applications, including:
- Public Safety: Ensures compliance with mask-wearing regulations in public places.
- Healthcare Monitoring: Helps in hospitals and clinics to enforce mask mandates.
- Smart Surveillance: Enhances security camera systems with real-time mask detection alerts.
- Workplace Compliance: Ensures employees adhere to mask guidelines in offices and industries.


Steps to Build

1. Data Collection : Use publicly available datasets of masked and unmasked faces.
2. Model Training : Train a CNN model using TensorFlow/Keras with labeled face mask data.
3. Preprocessing : Resize, normalize, and augment images to improve model accuracy.
4. Face Detection : Use OpenCV’s pre-trained Haar cascade or DNN face detector to locate faces.
5. Mask Classification : Apply the trained model to classify detected faces as "Mask" or "No Mask."
6. Real-time Detection : Integrate with a live camera feed to detect masks in real-time.


Work Flow

1. Input: The system captures video frames from a webcam or CCTV feed.
2. Processing: Face detection is performed using OpenCV, followed by classification using a CNN model.
3. Output: The system overlays "Mask" or "No Mask" labels on detected faces and provides alerts if needed.


Conclusion

This project presents an effective Face Mask Detection system that utilizes deep learning and computer vision techniques.By leveraging OpenCV and CNN models, the system can efficiently classify masked and unmasked faces in real-time. 
This technology can be widely implemented for public safety, healthcare, and surveillance, ensuring better enforcement of mask-wearing policies.
