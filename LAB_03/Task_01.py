import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# 1. DATASET PATH
# ==========================================

dataset_path = "animals"

cat_path = os.path.join(dataset_path, "cat")
dog_path = os.path.join(dataset_path, "dog")


# ==========================================
# 2. GET IMAGE FILES
# ==========================================

cat_images = os.listdir(cat_path)
dog_images = os.listdir(dog_path)

print("Number of cat images:", len(cat_images))
print("Number of dog images:", len(dog_images))

print("Total images:", len(cat_images) + len(dog_images))


# ==========================================
# 3. LOAD ONE ORIGINAL IMAGE
# ==========================================

cat_file = os.path.join(cat_path, cat_images[0])

image = Image.open(cat_file)

print("\nOriginal image size:", image.size)

plt.imshow(image)
plt.title("Original Cat Image")
plt.axis("off")
plt.show()


# ==========================================
# 4. CHECK CORRUPTED IMAGES
# ==========================================

corrupted = 0

for folder in [cat_path, dog_path]:

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            img = Image.open(path)
            img.verify()

        except:
            corrupted += 1
            print("Corrupted image:", path)


print("\nCorrupted images:", corrupted)


# ==========================================
# 5. RESIZE IMAGE
# ==========================================

resized_image = image.resize((128, 128))

print("Resized image size:", resized_image.size)

plt.imshow(resized_image)
plt.title("Resized Image - 128 x 128")
plt.axis("off")
plt.show()


# ==========================================
# 6. CONVERT IMAGE TO NUMPY ARRAY
# ==========================================

image_array = np.array(resized_image)

print("\nImage array shape:", image_array.shape)

print("Pixel values:")
print(image_array)


# ==========================================
# 7. NORMALIZE IMAGE
# ==========================================

normalized_image = image_array / 255.0

print("\nOriginal pixel range:")
print(image_array.min(), "to", image_array.max())

print("\nNormalized pixel range:")
print(normalized_image.min(), "to", normalized_image.max())


# ==========================================
# 8. ONE-HOT ENCODING
# ==========================================

labels = np.array(["cat", "dog"])

encoder = OneHotEncoder(sparse_output=False)

encoded_labels = encoder.fit_transform(
    labels.reshape(-1, 1)
)

print("\nOriginal labels:")
print(labels)

print("\nOne-hot encoded labels:")
print(encoded_labels)


# ==========================================
# 9. DISPLAY PREPROCESSED IMAGE
# ==========================================

plt.imshow(normalized_image)

plt.title("Preprocessed Cat Image")

plt.axis("off")

plt.show()

# ==========================================
# 9. DISPLAY ORIGINAL VS PREPROCESSED IMAGE
# ==========================================

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Preprocessed image
plt.subplot(1, 2, 2)
plt.imshow(normalized_image)
plt.title("Preprocessed Image")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\nTASK 01 COMPLETED!")



