import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize


# Путь к директории с изображениями
data_dir = 'D:\\Documents\\TeoriaNeironnok\\Kursach\\Datasets\\ultradataset'
images = []
labels = []

# Загрузка изображений и меток
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            image = load_img(img_path, target_size=(64, 64), color_mode='grayscale')
            image = img_to_array(image) / 255.0  # Нормализация
            images.append(image)
            labels.append(label)

# Преобразование списков в массивы NumPy
images = np.array(images)
labels = np.array(labels)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Преобразование строковых меток в числовые
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Создание объекта ImageDataGenerator для расширения данных
datagen = ImageDataGenerator(
    rotation_range=15,         # Случайный разворот изображений
    width_shift_range=0.1,     # Случайное смещение по ширине
    height_shift_range=0.1,    # Случайное смещение по высоте
    shear_range=0.1,           # Случайное сдвигание
    zoom_range=0.1,            # Случайное масштабирование
    fill_mode='nearest'        # Способ заполнения пустых зон
)

# Подгонка генератора к обучающим данным
datagen.fit(X_train)

# Извлечение 10 случайных изображений из генератора
sample_images, sample_labels = next(datagen.flow(X_train, y_train, batch_size=20))

# Установка количества строк и столбцов для отображения
num_images = 20
cols = 5
rows = num_images // cols

# Отображение изображений
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(sample_images[i])  # Если изображения в grayscale, используйте: cmap='gray'
    plt.title(f'Label: {sample_labels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Создание модели
model = keras.Sequential([
    layers.Conv2D(32, (4, 4), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(86, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(25, activation='softmax')  # 25 классов всех цифр
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Batch_size = 40
Epochs = len(X_train) // Batch_size

print(f"Эпохи равныыыыыыыыыыыыыыыыыыыыыыыы {Epochs}")

# Обучение модели с использованием генератора
model.fit(datagen.flow(X_train, y_train_encoded, batch_size=Batch_size), 
          epochs=Epochs, 
          validation_data=(X_test, y_test_encoded))

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc:.4f}')

# Сохранение модели
#model.save('digit_recognition_model_Ultra_s_generatorom.h5')

def plot_random_predictions(X_test, y_test, y_test_encoded, model, encoder, n=10):
    random_indices = np.random.choice(len(X_test), n, replace=False)
    
    plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(64, 64), cmap='gray')
        plt.axis('off')
        
        # Предсказание на основе модели
        prediction = model.predict(X_test[idx].reshape(1, 64, 64, 1))
        predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]
        
        # Отображение реальной и предсказанной меток
        plt.title(f'True: {y_test[idx]}\nPred: {predicted_label}')
    
    plt.tight_layout()
    plt.show()

# Вызов функции для отображения 10 случайных изображений
plot_random_predictions(X_test, y_test, y_test_encoded, model, encoder)

# Получение предсказаний от модели
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Создание DataFrame с реальными и предсказанными метками
results_df = pd.DataFrame({
    'True Labels': y_test,
    'Predicted Labels': predicted_labels
})

# Отображение таблицы
print(results_df)
#joblib.dump(encoder, 'label_encoder_Ultra.pkl')

# 1. Classification Report
print("\nClassification Report:")
print(classification_report(y_test_encoded, predicted_labels, 
                           target_names=encoder.classes_))

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_encoded, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.classes_, 
            yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 3. ROC Curve (для многоклассовой классификации)
y_test_bin = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded))
pred_proba = model.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(encoder.classes_)

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(encoder.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.show()

# 4. Precision-Recall Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], pred_proba[:, i])
    plt.plot(recall, precision, lw=2,
             label='Class {0} (AP = {1:0.2f})'
             ''.format(encoder.classes_[i], auc(recall, precision)))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.show()

# 5. Top-K Accuracy
def top_k_accuracy(y_true, y_pred_proba, k=3):
    top_k = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            correct += 1
    return correct / len(y_true)

for k in [2, 3, 5]:
    print(f"Top-{k} Accuracy: {top_k_accuracy(y_test_encoded, pred_proba, k):.4f}")

# 6. Class Distribution Comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=y_test)
plt.title('True Class Distribution')
plt.subplot(1, 2, 2)
sns.countplot(x=predicted_labels)
plt.title('Predicted Class Distribution')
plt.tight_layout()
plt.show()

# 7. Error Analysis
errors = (predicted_labels != y_test_encoded)
error_indices = np.where(errors)[0]

plt.figure(figsize=(15, 8))
for i, idx in enumerate(error_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[idx].reshape(64, 64), cmap='gray')
    plt.title(f'True: {y_test[idx]}\nPred: {encoder.classes_[predicted_labels[idx]]}')
    plt.axis('off')
plt.suptitle('Примеры ошибок классификации')
plt.tight_layout()
plt.show()