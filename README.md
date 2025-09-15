# 🧠 Face Mask Detection
A real-time AI system that detects face mask usage using CNN and OpenCV.


## 📸 Demo

![Demo Screenshot](D:\my learning\python\demo_withmask.png.png)  
*Live detection with confidence scores and bounding boxes.*


## 🚀 Features
- Real-time webcam detection
- CNN model trained on custom dataset
- Confidence-based classification: Mask / No Mask / Uncertain
- Accuracy visualization with Matplotlib
- Modular code for training and inference


## 📊 Model Performance
- Training Accuracy: ~92%
- Validation Accuracy: ~85%
- Confidence Thresholds:
  - Mask: prediction < 0.4
  - No Mask: prediction > 0.6
  - Uncertain: 0.4 ≤ prediction ≤ 0.6


## 🧰 Tech Stack

| Component      		   | Description                     	   		   |
|-----------------|------------------------------- |
| TensorFlow/Keras | CNN model training and evaluation |
| OpenCV         		   | Webcam capture and face detection |
| Haar Cascade  		   | Face localization                                   |
| Matplotlib   		   | Accuracy visualization                       |
| Python        		   | Core scripting and logic                     |


## 📂 Project Structure
face-mask-detection/ 
├── train/	 				# Training images (with_mask, without_mask)
├── validation/	 			# Validation images
├── best_mask_model.h5 		# Saved model (HDF5 format) 
├── facemaskdetection.py 	# Main script 
├── README.md 					# Project documentation 
└── demo_image.png 			# Screenshot of live detection



## 🧑‍💻 How to Run
1. Clone the repo  
2. Install dependencies  
3. Run `facemaskdetection.py`  
4. Press `q` to exit webcam window


## Install Dependencies

pip install tensorflow opencv-python matplotlib pillow


## Train the Model (Optional)

Uncomment the training block in facemaskdetection.py and run:

**  python facemaskdetection.py





## 📁 Dataset Format
train/
├── with_mask/
└── without_mask/

validation/
├── with_mask/
└── without_mask/


📈 Future Improvements
🔄 Transfer learning with MobileNetV2

📱 TensorFlow Lite conversion for mobile deployment

🖼️ Grad-CAM visualization for model interpretability

🧪 Confusion matrix and precision-recall analysis



## 👨‍⚕️ Author
**Roop** — Medical Officer & Tech Enthusiast  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)  
💻 [Portfolio](https://yourportfolio.com)






