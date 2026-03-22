import urllib.request
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

print("Downloading SAM model... (this is large ~2.5GB)")

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
urllib.request.urlretrieve(url, "models/sam_vit_h_4b8939.pth")

print("✅ Download complete!")