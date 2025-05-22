import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib  # Для сохранения и загрузки LabelEncoder
import cv2
import sys

# Загрузка обученной модели
try:
    model = tf.keras.models.load_model('digit_recognition_model_ultra_s_generatorom.h5')
except Exception as e:
    tk.messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")
    sys.exit(1)

# Загрузка LabelEncoder
try:
    encoder = joblib.load('label_encoder_ultra.pkl')  # Сохраните и загрузите LabelEncoder
except Exception as e:
    tk.messagebox.showerror("Ошибка", f"Не удалось загрузить label encoder: {str(e)}")
    sys.exit(1)

# Функция для предсказания цифры
def predict_digit(image_array):
    # Проверяем количество каналов в изображении
    if image_array.ndim == 2:
        gray = image_array
    elif image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Некорректное количество каналов в изображении")

    # # Применяем пороговое преобразование для получения черной цифры на белом фоне
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # # Находим контуры
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if len(contours) == 0:
    #     return None

    # # Находим границы цифры
    # x, y, w, h = cv2.boundingRect(contours[0])
    # digit = thresh[y:y+h, x:x+w]
    
    # # Добавляем отступы для центрирования
    # padded = cv2.copyMakeBorder(digit, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # centered_image = cv2.resize(padded, (64, 64))
    
    # # Нормализация
    # img_array = centered_image / 255.0
    # img_array = img_array.reshape(1, 64, 64, 1)
    gray = gray.reshape(1, 64, 64, 1)

    # Отображение изображения
    #plt.imshow(centered_image, cmap='gray')
    #plt.axis('off')
    #plt.show()

    # Предсказание
    predictions = model.predict(gray)
    predicted_digit = np.argmax(predictions)
    return predicted_digit

def segment_digits(image_path, output_size=(64, 64)):
    # 1. Загрузка изображения и преобразование в оттенки серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Бинаризация: черные цифры на белом фоне
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Удаление мелкого шума (опционально)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Поиск контуров цифр
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Выделение и сортировка bounding box
    digit_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        digit_boxes.append((x, y, w, h))
    digit_boxes.sort(key=lambda box: box[0])  # Сортировка слева направо
    
    # 6. Обрезка, центрирование и масштабирование цифр
    digits = []
    for x, y, w, h in digit_boxes:
        digit_img = binary[y:y+h, x:x+w]
        
        # Создание белого фона с чёрной цифрой
        digit_processed = 255 - digit_img  # Инверсия: чёрная цифра на белом
        
        # Добавление отступов для сохранения пропорций
        height, width = digit_processed.shape
        scale = 0.8 * min(output_size[0]/height, output_size[1]/width)
        new_w, new_h = int(width * scale), int(height * scale)
        resized = cv2.resize(digit_processed, (new_w, new_h))
        
        # Помещаем цифру в центр изображения 64x64
        canvas = np.ones(output_size, dtype=np.uint8) * 255  # Белый фон
        x_offset = (output_size[0] - new_w) // 2
        y_offset = (output_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        digits.append(canvas)
    
    return digits

#Функция для показа предсказанных цифр
def show_digits_with_predictions(digits, predictions, encoder, ncols=4):
    """
    Отображает все распознанные цифры с подписями в grid-формате.
    
    Параметры:
        digits (list): Список изображений цифр (numpy arrays).
        predictions (list): Список предсказанных меток (числа).
        encoder: Ваш LabelEncoder для преобразования чисел в символы.
        ncols (int): Количество колонок в grid-е.
    """
    n = len(digits)
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2 * nrows))
    if n == 1:
        axes = np.array([axes])  # Чтобы избежать ошибок с 1 цифрой
    axes = axes.flatten()
    
    for i, (digit, pred) in enumerate(zip(digits, predictions)):
        ax = axes[i]
        ax.imshow(digit, cmap='gray')
        predicted_label = encoder.inverse_transform([pred])[0]
        ax.set_title(f"Pred: {predicted_label}", fontsize=10)
        ax.axis('off')
    
    # Скрываем пустые subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.canvas.manager.set_window_title("Предсказанные цифры")
    plt.tight_layout()
    plt.show()

# Функция для сохранения и обработки нарисованного изображения
def save_and_predict():
    #Проверка на пустой холст
    if len(canvas.find_all()) == 0:  # Функция проверки (например, по наличию пикселей)
        tk.messagebox.showwarning("Ошибка", "Нарисуйте цифру(ы) перед распознаванием!")
        return

    # Сохраняем изображение в файл
    file_path = "drawn_digit.png"
    image.save(file_path)
    
    # Предобрабатываем изображение для нейросети
    #img = image.resize((64, 64)).convert('L')  # Изменяем размер и преобразуем в чёрно-белое
    #img_array = np.array(img)

    digits = segment_digits(file_path)
    for i, digit in enumerate(digits):
        cv2.imwrite(f"digit_{i}.png", digit)

    predictions = [predict_digit(d) for d in digits]  # Предсказания для каждой цифры
    show_digits_with_predictions(digits, predictions, encoder, ncols=4)

    #Распознаём цифру
    #for i in digits:
       # digit = predict_digit(i)
       # predicted_label = encoder.inverse_transform([digit])[0]  # Преобразуем число в строку
       # result_label.config(text=f"Распознанная цифра: {predicted_label}")


# Функция для рисования
def paint(event):
    x, y = event.x, event.y
    r = 12  # Радиус кисти
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
    draw.ellipse((x-r, y-r, x+r, y+r), fill='black')

# Функция для очистки canvas
def clear_canvas():
    canvas.delete("all")  # Очищаем canvas
    draw.rectangle((0, 0, 800, 450), fill="white")  # Очищаем изображение

# Создание графического интерфейса
root = tk.Tk()
root.title("Рисование и распознавание цифр")

# Настраиваем канвас для рисования
canvas = tk.Canvas(root, width=800, height=450, bg='white')
canvas.pack(pady=10)
canvas.bind("<B1-Motion>", paint)

clear_button = tk.Button(root, text="Очистить", command=clear_canvas)
clear_button.pack(pady=5)

# Создаем новое изображение для рисования
image = Image.new("RGB", (800, 450), "white")
draw = ImageDraw.Draw(image)

# Кнопка для сохранения и распознавания
predict_button = tk.Button(root, text="Распознать рисунок", command=save_and_predict)
predict_button.pack(pady=10)

# Метка для вывода результата
#result_label = tk.Label(root, text="Распознанная цифра: ", font=("Arial", 16))
#result_label.pack(pady=10)

# Запуск основного цикла
root.mainloop()