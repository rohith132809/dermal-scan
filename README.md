# DermalScan – AI Facial Skin Aging Detection App

DermalScan is an AI-powered deep learning application designed to detect and classify facial skin aging signs from images.  
The system leverages **transfer learning with EfficientNetB0**, along with OpenCV-based face detection and a user-friendly web interface for real-time inference and visualization.

---

## 🧠 Project Statement

The objective of DermalScan is to develop an AI-driven system capable of identifying and classifying facial aging indicators such as:

- Wrinkles  
- Dark spots  
- Puffy eyes  
- Clear skin  

The application integrates:
- OpenCV-based face detection using Haar Cascades  
- Custom image preprocessing and data augmentation  
- Deep learning models built with TensorFlow/Keras  
- A web-based frontend for user interaction and result visualization  

---

## 🎯 Expected Outcomes

1. **Detection & Localization** – Detect facial regions and highlight aging signs  
2. **Classification** – Wrinkles, Dark Spots, Puffy Eyes, Clear Skin  
3. **Model Performance** – Robust CNN with high accuracy  
4. **User Interface** – Image upload with annotated predictions  
5. **Backend Integration** – Real-time inference and logging  

---

## 📦 Project Modules

### Module 1: Dataset Setup and Image Labeling
Prepared a clean and balanced dataset with four classes:
```
wrinkles/
dark_spots/
puffy_eyes/
clear_skin/
```

### Module 2: Image Preprocessing and Augmentation
- Resize to 224×224  
- Normalize to [0,1]  
- Augmentation using ImageDataGenerator  

### Module 3: Model Training
Models used:
- EfficientNet-B0  
- DenseNet121  
- MobileNetV2  

### Model Accuracy
| Model | Accuracy |
|------|----------|
| MobileNetV2 | 86.07% |
| DenseNet121 | 87.34% |
| EfficientNet-B0 | 83.54% |

### Module 4: Face Detection Pipeline
- OpenCV Haar Cascade
- Bounding boxes + confidence scores

### Module 5: Frontend
- Streamlit-based UI
- Image upload and annotated preview

### Module 6: Backend Pipeline
- Modular inference
- Logging and optimization

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/DermalScan.git
cd DermalScan
pip install -r requirements.txt
python backend.py
streamlit run app.py
```

---

## 📌 Technologies Used
Python, TensorFlow/Keras, OpenCV, Streamlit, NumPy, Matplotlib

---

## 👤 Author
**Rohith Lalam**  
AI / ML Engineer

---

## 📄 License
Academic and educational use only.
