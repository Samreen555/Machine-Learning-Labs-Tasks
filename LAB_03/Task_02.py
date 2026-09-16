import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ==========================================
# 1. DATASET PATH
# ==========================================

dataset_path = "animals"

cat_path = os.path.join(dataset_path, "cat")
dog_path = os.path.join(dataset_path, "dog")


# ==========================================
# 2. LOAD IMAGES
# ==========================================

images = []
original_images = []

# Load 10 cat images
for file in os.listdir(cat_path)[:10]:

    path = os.path.join(cat_path, file)

    try:
        # Load original image
        original = Image.open(path).convert("RGB")

        # Save original image
        original_images.append(original)

        # Resize a copy for processing
        resized = original.resize((32, 32))

        # Convert resized image to NumPy array
        resized = np.array(resized)

        images.append(resized)

    except:
        print("Could not load:", path)


# Load 10 dog images
for file in os.listdir(dog_path)[:10]:

    path = os.path.join(dog_path, file)

    try:
        # Load original image
        original = Image.open(path).convert("RGB")

        # Save original image
        original_images.append(original)

        # Resize a copy for processing
        resized = original.resize((32, 32))

        # Convert resized image to NumPy array
        resized = np.array(resized)

        images.append(resized)

    except:
        print("Could not load:", path)


# ==========================================
# 3. CONVERT TO NUMPY ARRAY
# ==========================================

images = np.array(images)

print("Image shape:")
print(images.shape)


# ==========================================
# 4. FLATTEN PIXELS
# ==========================================

pixels = images.reshape(images.shape[0], -1)

print("\nPixel data shape:")
print(pixels.shape)


# ==========================================
# 5. ORIGINAL PIXEL DISTRIBUTION
# ==========================================

plt.hist(pixels.flatten(), bins=50)

plt.title("Original Pixel Distribution")

plt.xlabel("Pixel Value")

plt.ylabel("Frequency")

plt.show()


# ==========================================
# 6. STANDARD SCALER
# ==========================================

standard_scaler = StandardScaler()

standard_data = standard_scaler.fit_transform(pixels)

print("\nStandardScaler")

print("Mean:", standard_data.mean())

print("Standard deviation:", standard_data.std())


# ==========================================
# 7. STANDARD SCALER DISTRIBUTION
# ==========================================

plt.hist(standard_data.flatten(), bins=50)

plt.title("After StandardScaler")

plt.xlabel("Scaled Pixel Value")

plt.ylabel("Frequency")

plt.show()


# ==========================================
# 8. STANDARD SCALER IMAGE
# ==========================================

standard_image = standard_data[0].reshape(32, 32, 3)

# Convert StandardScaler values to 0-1
# only for displaying the image
standard_display = (
    standard_image - standard_image.min()
) / (
    standard_image.max() - standard_image.min()
)


# ==========================================
# 9. ORIGINAL VS STANDARD SCALER
# ==========================================

plt.figure(figsize=(12, 5))


# Actual original image
plt.subplot(1, 2, 1)

plt.imshow(original_images[0])

plt.title("Original Image")

plt.axis("off")


# StandardScaler image
plt.subplot(1, 2, 2)

plt.imshow(standard_display)

plt.title("After StandardScaler")

plt.axis("off")


plt.tight_layout()

plt.show()


# ==========================================
# 10. MIN-MAX SCALER
# ==========================================

minmax_scaler = MinMaxScaler()

minmax_data = minmax_scaler.fit_transform(pixels)

print("\nMinMaxScaler")

print("Minimum:", minmax_data.min())

print("Maximum:", minmax_data.max())


# ==========================================
# 11. MIN-MAX DISTRIBUTION
# ==========================================

plt.hist(minmax_data.flatten(), bins=50)

plt.title("After MinMaxScaler")

plt.xlabel("Scaled Pixel Value")

plt.ylabel("Frequency")

plt.show()


# ==========================================
# 12. MIN-MAX IMAGE
# ==========================================

minmax_image = minmax_data[0].reshape(32, 32, 3)


# ==========================================
# 13. ORIGINAL VS MIN-MAX
# ==========================================

plt.figure(figsize=(12, 5))


# Actual original image
plt.subplot(1, 2, 1)

plt.imshow(original_images[0])

plt.title("Original Image")

plt.axis("off")


# MinMaxScaler image
plt.subplot(1, 2, 2)

plt.imshow(minmax_image)

plt.title("After MinMaxScaler")

plt.axis("off")


plt.tight_layout()

plt.show()


# ==========================================
# 14. COMPARISON
# ==========================================

print("\n========== COMPARISON ==========")

print("""
Original pixels:
Values are between 0 and 255.

StandardScaler:
Mean becomes approximately 0
and standard deviation becomes approximately 1.

MinMaxScaler:
Values are scaled between 0 and 1.
""")


print("\nTASK 02 COMPLETED!")