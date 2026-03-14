# CNN-Based Face Recognition System

This project implements a **Convolutional Neural Network (CNN) based Face Recognition System** using **Python, TensorFlow/Keras, and OpenCV**.  
The system detects faces using a webcam and recognizes individuals based on a trained CNN model.

This project was developed as part of an **internship case study on CNN-Based Face Recognition Systems**.

---

# Project Overview

Face recognition is a biometric technology used to identify or verify a person using facial features extracted from images.  
This project demonstrates how **deep learning (CNNs)** can be used to perform real-time face recognition.

The system performs the following tasks:

1. Collect facial images using a webcam
2. Train a CNN model on the dataset
3. Detect faces in real-time
4. Recognize individuals based on trained data

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

---

# Project Structure
cnn-face-recognition-system
│
├── dataset/ # Face dataset
│
├── collect_faces.py # Script to collect face images
├── train_model.py # CNN model training script
├── recognize_face.py # Real-time face recognition
│
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore



---

# CNN Architecture Used

The Convolutional Neural Network used in this project consists of:

- Input Layer (100x100 grayscale image)
- Convolution Layer (32 filters)
- Max Pooling Layer
- Convolution Layer (64 filters)
- Max Pooling Layer
- Flatten Layer
- Dense Layer (128 neurons)
- Output Layer (Softmax classifier)

---

# Installation

Clone the repository:
git clone https://github.com/YOUR_USERNAME/cnn-face-recognition-system.git
cd cnn-face-recognition-system



Install dependencies:
pip install -r requirements.txt



---

# How to Run the Project

### 1. Collect Face Dataset
python collect_faces.py



Enter your name and capture face images using the webcam.

---

### 2. Train the CNN Model
python train_model.py



This will train the CNN model and save it as:
face_model.h5



---

### 3. Run Face Recognition
python recognize_face.py



The webcam will open and recognize faces in real time.

---

# Applications

Face recognition systems are widely used in:

- Security and surveillance systems
- Smartphone authentication
- Attendance systems
- Identity verification
- Access control systems

---

# Future Improvements

Possible enhancements to this project include:

- Using pre-trained models like **FaceNet**
- Improving accuracy using larger datasets
- Adding an **Unknown face detection threshold**
- Building a **web interface using Flask**
- Storing recognition logs in a database

---

# Author

Laxman Mudigonda  
B.Tech Computer Science Engineering  
GITAM University