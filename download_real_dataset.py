import os
import cv2
import numpy as np
import urllib.request

def download_specimens():
    # Direct links to public domain Government Specimen IDs
    urls =[
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/California_Driver_License_-_Sample.svg/800px-California_Driver_License_-_Sample.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/New_York_State_Driver_License_Sample.png/800px-New_York_State_Driver_License_Sample.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/PRT_Passport.jpg/800px-PRT_Passport.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/UK_driving_licence.png/800px-UK_driving_licence.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Personalausweis_Muster.jpg/800px-Personalausweis_Muster.jpg"
    ]
    
    os.makedirs("temp_specimens", exist_ok=True)
    os.makedirs("dataset/genuine", exist_ok=True)
    os.makedirs("dataset/counterfeit", exist_ok=True)
    
    downloaded_paths =[]
    print("Downloading Real Public Domain Specimen IDs...")
    for i, url in enumerate(urls):
        path = f"temp_specimens/specimen_{i}.jpg"
        urllib.request.urlretrieve(url, path)
        downloaded_paths.append(path)
        print(f"Downloaded Specimen {i+1}/5")
        
    return downloaded_paths

def generate_dataset(base_images):
    print("\nMultiplying into 100 Hybrid Dataset Images...")
    gen_count = 0
    fake_count = 0
    
    for img_path in base_images:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 500))
        
        # 1. Generate 10 Genuine Variations per image (Simulating different smartphone cameras)
        for i in range(10):
            # Mild brightness/contrast changes for real captures
            alpha = np.random.uniform(0.9, 1.1) # Contrast
            beta = np.random.randint(-10, 10)   # Brightness
            genuine_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            cv2.imwrite(f"dataset/genuine/real_specimen_{gen_count}.jpg", genuine_aug)
            gen_count += 1
            
        # 2. Generate 10 Counterfeit Variations per image (Simulating fake printers & Photoshop)
        for i in range(10):
            fake_aug = img.copy()
            
            # Add heavy blur to simulate cheap printing/scanning
            fake_aug = cv2.GaussianBlur(fake_aug, (7, 7), 0)
            
            # Add digital compression artifacts (JPG quality loss)
            encode_param =[int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(10, 40)]
            _, encimg = cv2.imencode('.jpg', fake_aug, encode_param)
            fake_aug = cv2.imdecode(encimg, 1)
            
            # Add malicious pixel noise
            noise = np.random.normal(0, 15, fake_aug.shape).astype(np.uint8)
            fake_aug = cv2.add(fake_aug, noise)
            
            cv2.imwrite(f"dataset/counterfeit/fake_specimen_{fake_count}.jpg", fake_aug)
            fake_count += 1

    print(f"\nSuccess! Generated {gen_count} Genuine IDs and {fake_count} Counterfeit IDs.")
    print("Check your 'dataset/genuine' and 'dataset/counterfeit' folders!")

if __name__ == "__main__":
    downloaded_imgs = download_specimens()
    generate_dataset(downloaded_imgs)