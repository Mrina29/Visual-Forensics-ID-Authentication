import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from skimage.feature import local_binary_pattern

# Load Pre-trained ResNet for Holograms/Seals
weights = models.ResNet18_Weights.DEFAULT
cnn_model = models.resnet18(weights=weights)
cnn_model = torch.nn.Sequential(*(list(cnn_model.children())[:-1]))
cnn_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_and_align(image):
    # Converts raw image into a flat, 800x500 standard document
    orig_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    img_area = image.shape[0] * image.shape[1]
    screenCnt = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Ensure the contour has 4 points AND covers at least 30% of the image area
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * img_area:
            screenCnt = approx
            break

    # If it can't find the outer edge, just return the resized original image
    if screenCnt is None:
        return cv2.resize(orig_copy, (800, 500))

    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig_copy, M, (maxWidth, maxHeight))
    
    return cv2.resize(warped, (800, 500))

def extract_rois(aligned_img):
    # Slicing the 800x500 image into logical regions
    return {
        "text_region": aligned_img[100:300, 300:700],      
        "background_texture": aligned_img[50:150, 50:200], 
        "seal_hologram": aligned_img[350:480, 650:780]     
    }

def extract_lbp_features(image_roi, P=8, R=1):
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_cnn_features(image_roi):
    input_tensor = preprocess(image_roi).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(input_tensor)
    return features.numpy().flatten()

def get_fused_vector(rois):
    bg_features = extract_lbp_features(rois["background_texture"])
    font_features = extract_lbp_features(rois["text_region"])
    seal_features = extract_cnn_features(rois["seal_hologram"])
    return np.hstack([bg_features, font_features, seal_features])