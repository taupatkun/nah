import streamlit as st
from PIL import Image
import torch
import pickle
import torchvision.transforms as transforms

# Set the title of the app
st.title("AI Model Image Prediction App")

# Load the .pkl model
@st.cache_resource
def load_model():
    with open("Skin_disease.pkl", "rb") as file:
        model = pickle.load(file)
    model.eval()  # Set the model to evaluation mode (for PyTorch models)
    return model

model = load_model()

# Define image transformations (adjust based on your model training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Define class labels (customize based on your model)
LABELS = ["Class 1", "Class 2", "Class 3"]

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Apply the transformations
    img_tensor = transform(image).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        prediction = LABELS[predicted.item()]

    # Display the prediction
    st.write(f"Prediction: **{prediction}**")
else:
    st.write("Please upload an image to get a prediction.")