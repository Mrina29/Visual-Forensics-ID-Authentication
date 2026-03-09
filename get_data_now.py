import os
import cv2
import numpy as np
import requests

def get_base_images():
    urls =[
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/California_Driver_License_-_Sample.svg/800px-California_Driver_License_-_Sample.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/New_York_State_Driver_License_Sample.png/800px-New_York_State_Driver_License_Sample.png"
    ]
    
    os.makedirs("temp_specimens", exist_ok=True)
    os.makedirs(os.path.join("dataset", "genuine"), exist_ok=True)
    os.makedirs(os.path.join("dataset", "counterfeit"), exist_ok=True)
    
    downloaded_paths =[]
    print("Attempting to download Real Specimen IDs...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # Try downloading
    for i, url in enumerate(urls):
        path = f"temp_specimens/specimen_{i}.jpg"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                downloaded_paths.append(path)
                print(f"Downloaded Specimen {i+1} successfully!")
        except Exception as e:
            pass # Ignore the error and let the fallback handle it
            
    # THE ULTIMATE FALLBACK: If Wikipedia blocks you, generate local realistic bases instantly!
    if len(downloaded_paths) == 0:
        print("Network blocked by Wikipedia. Generating realistic local base templates instead...")
        for i in range(2):
            path = f"temp_specimens/fallback_{i}.jpg"
            img = np.ones((500, 800, 3), dtype=np.uint8) * 240
            # Draw realistic complex guilloche background
            for j in range(0, 800, 15):
                cv2.line(img, (j, 0), (j+50, 500), (200, 220, 200), 1)
            cv2.rectangle(img, (50, 100), (250, 350), (200, 150, 150), -1) # Photo box
            cv2.putText(img, f"OFFICIAL ID SPECIMEN {i}", (280, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3)
            cv2.putText(img, "Name: JOHN DOE", (280, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
            cv2.putText(img, "DOB: 01/01/1990", (280, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
            cv2.circle(img, (680, 400), 60, (50, 50, 180), -1) # Hologram/Seal
            cv2.circle(img, (680, 400), 50, (100, 100, 200), 2)
            cv2.imwrite(path, img)
            downloaded_paths.append(path)
            
    return downloaded_paths

def generate_hybrid_dataset(base_images):
    print("\nMultiplying images into 100 variations for Machine Learning...")
    gen_count = 0
    fake_count = 0
    
    for img_path in base_images:
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (800, 500))
        
        # 1. Genuine Variations (High quality, slight lighting changes)
        for i in range(25):
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.randint(-15, 15)
            genuine_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            cv2.imwrite(f"dataset/genuine/real_img_{gen_count}.jpg", genuine_aug)
            gen_count += 1
            
        # 2. Counterfeit Variations (Blur, compression, printer noise)
        for i in range(25):
            fake_aug = img.copy()
            # Apply blur to simulate cheap printing
            fake_aug = cv2.GaussianBlur(fake_aug, (9, 9), 0)
            
            # Apply JPG compression artifacts
            encode_param =[int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(5, 30)]
            _, encimg = cv2.imencode('.jpg', fake_aug, encode_param)
            fake_aug = cv2.imdecode(encimg, 1)
            
            # Add severe pixel noise
            noise = np.random.normal(0, 20, fake_aug.shape).astype(np.float32)
            fake_aug = np.clip(fake_aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(f"dataset/counterfeit/fake_img_{fake_count}.jpg", fake_aug)
            fake_count += 1

    print(f"\nSUCCESS! Generated {gen_count} Genuine and {fake_count} Counterfeit IDs.")
    print("Your dataset folders are now full and ready!")

if __name__ == "__main__":
    base_imgs = get_base_images()
    generate_hybrid_dataset(base_imgs)