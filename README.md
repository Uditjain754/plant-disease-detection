# plant-disease-detection
Plant Disease Detection is a deep learning–powered project that classifies plant leaf images to identify diseases accurately. Built using image processing and trained models, this project aims to assist in early diagnosis, reduce crop loss, and support smart agriculture practices.

<img width="685" height="679" alt="image" src="https://github.com/user-attachments/assets/a6f3bb49-3359-485c-aafa-10e6d46c07ed" />


✨ Features
Instant Disease Detection: Upload an image of a plant leaf and get immediate diagnosis results
Comprehensive Diagnosis: Provides detailed information on detected diseases, including causes, treatments, and prevention
User-Friendly Interface: Clean, intuitive UI with image previews and visualization of results
Multiple Plant Support: Currently supports tomatoes, potatoes, and bell peppers
High Accuracy: 96.5% accuracy on test datasets
Example Images: Try the application with pre-loaded example images


🧪 Supported Plant Diseases
The model can currently identify the following plants and diseases:

Tomato:
Healthy
Bacterial Spot
Early Blight
Late Blight
Leaf Mold
Septoria Leaf Spot
Spider Mites
Target Spot
Yellow Leaf Curl Virus
Mosaic Virus


Potato:
Healthy
Early Blight
Late Blight

Bell Pepper:
Healthy
Bacterial Spot

🔧 Model Architecture
The application uses a custom CNN architecture with the following components:

5 convolutional blocks with batch normalization and ReLU activation
Global Average Pooling
Fully connected layers with dropout for regularization
Trained on the PlantVillage dataset with data augmentation techniques
Achieves 96.5% accuracy on the test set
