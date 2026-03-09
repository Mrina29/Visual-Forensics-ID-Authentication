import cv2
import numpy as np
import os

def create_synthetic_id(is_genuine, filename):
    # Create a blank white ID card base (800x500)
    img = np.ones((500, 800, 3), dtype=np.uint8) * 255
    
    # 1. Background Texture (Guilloche Simulation)
    for i in range(0, 800, 20):
        color = (200, 255, 200) if is_genuine else (180, 200, 180) # Fakes have duller colors
        thickness = 1 if is_genuine else 2 # Fakes bleed ink
        cv2.line(img, (i, 0), (i+50, 500), color, thickness)
        cv2.line(img, (0, i//2), (800, i//2+50), color, thickness)

    # 2. Add ID Photo placeholder (Blue box)
    cv2.rectangle(img, (50, 100), (250, 350), (250, 150, 100), -1)
    
    # 3. Add Text (Name, DOB, ID Number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "REPUBLIC OF DATA", (300, 100), font, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "Name: JOHN DOE", (300, 200), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "DOB: 01-01-1990", (300, 250), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "ID: XJ-90210-44", (300, 300), font, 1, (0, 0, 0), 2)
    
    # 4. Add "Hologram/Seal" (Red Circle)
    cv2.circle(img, (650, 400), 60, (50, 50, 200), -1)
    cv2.circle(img, (650, 400), 50, (100, 100, 250), 2)
    
    # APPLY COUNTERFEIT EFFECTS (The differences your AI will learn)
    if not is_genuine:
        # Fake IDs usually have JPG compression artifacts, blur, and color noise from cheap printers
        img = cv2.GaussianBlur(img, (5, 5), 0) # Blurry text
        noise = np.random.randint(0, 30, (500, 800, 3), dtype='uint8')
        img = cv2.add(img, noise) # Grainy texture
    
    cv2.imwrite(filename, img)

print("Generating 40 Genuine ID samples...")
for i in range(40):
    create_synthetic_id(True, f"dataset/genuine/gen_{i}.jpg")

print("Generating 40 Counterfeit ID samples...")
for i in range(40):
    create_synthetic_id(False, f"dataset/counterfeit/fake_{i}.jpg")

# Create two test images to use during your presentation
create_synthetic_id(True, "sample_test_images/test_genuine.jpg")
create_synthetic_id(False, "sample_test_images/test_fake.jpg")

print("Dataset generated successfully!")