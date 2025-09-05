# Mobyglobal: A real-time whale detection network.

[Project Board](https://isef.net/project/robo046t-mobyglobal-a-real-time-whale-detection-network)  
  
[Video Demo](https://www.youtube.com/watch?v=JJYqr3itXvQ)

## Project Structure

### Android App (`android_src/`)
Contains the Android application code, including:
- Gradle build configuration
- Java source code under `app/src/main/java/com/lirawjani/`
- Resource files (layouts, drawables, fonts) under `app/src/main/res/`

### Buoy Hardware (`buoy_src/`)
Contains the buoy-related code and firmware:
- Python processing scripts (for raspberry pi)
- Arduino firmware for microphone (`Micophone/Micophone.ino`)
- Additional sketch files

### ML Model (`Moby_src/`)
Contains machine learning model files and training data:
- Trained model files (`.h5`, `.pt`)
- Training data (`.aiff`, `.wav` files)
- Jupyter notebooks for model development
- Spectrogram data

### Server (`server_src/`)
Contains the web server implementation:
- Flask application (`app.py`) and frontend
- Database models and processing logic
- HTML templates
- Static assets
- API testing files
- Configuration files (requirements.txt, runtime.txt, vercel.json)