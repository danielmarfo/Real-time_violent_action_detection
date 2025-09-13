# Real-Time Violence Action Detection

This repository contains the implementation of my Final Year Project: a real-time AI system for detecting violent actions in surveillance video streams.

The system uses a MobileNetV2 + BiLSTM model trained on the Real Life Violence Situations dataset to classify video clips as violent or non-violent. A Flask web application provides live webcam monitoring, real-time detection overlays, and audio alerts.

## Features
Real-time violence detection from live video streams.
Lightweight deep learning model (MobileNetV2 + BiLSTM).
Web-based interface built with Flask.
Visual and audio alerts on detection.

## Demo
<img width="1075" height="804" alt="Dashboard" src="https://github.com/user-attachments/assets/8b6405af-a8a4-4f88-bb55-c8f6bb3f4f9b" />

<img width="1075" height="804" alt="Handshaking detected as non-violent" src="https://github.com/user-attachments/assets/f202aaa6-8dc3-41b0-a7c7-40d72f623992" />
**Handshaking detected as non-violent**

<img width="1075" height="804" alt="Pushing detected as violent" src="https://github.com/user-attachments/assets/8681c05f-5d4d-4ad3-a30c-474721ba7255" />
**Pushing detected as violent**


## Install dependencies:
pip install -r requirements.txt
## Run the application:
python app.py
Open the web interface at:
http://127.0.0.1:5000/

**Link to kaggle(training and results):** https://www.kaggle.com/code/danielmarfo/violence-detection-model-mobilenet-bi-lstm

 ## Acknowledgements: 
 https://github.com/abduulrahmankhalid/Real-Time-Violence-Detection
