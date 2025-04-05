# 🧠 ASL Alphabet Recognition (A, B, C) with Simulation

This project is a real-time American Sign Language (ASL) recognition system that detects and classifies the letters **A, B, and C** using a custom-built dataset, MediaPipe for hand tracking, and a deep learning model trained from scratch. A simulation environment is also created using **Pygame**, where detected letters are visualized in an interactive manner.

---

## 🔍 Features

- ✋ Real-time hand tracking using **MediaPipe**
- 🧠 Custom **CNN** model trained on a self-collected dataset
- 🧪 Accurate recognition of **A**, **B**, and **C**
- 🕹️ Visual simulation using **Pygame** to represent detected signs interactively
- 📂 Organized and modular codebase for easy understanding and extension

## 🧰 Tech Stack

- **Python**
- **MediaPipe** – for hand landmarks extraction
- **TensorFlow/Keras** – for training and inference of classification model
- **OpenCV** – for real-time video feed
- **Pygame** – for the simulation interface

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/diyaaahh/SignLang
cd SignLang
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the files
```bash
python3 test.py
python3 simulation.py
