import gradio as gr
from transformers import pipeline

# Загружаем готовую модель для классификации изображений
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Функция обработки изображения
def predict(image):
    results = classifier(image)
    # Берём топ-1 результат
    top_result = results[0]
    label = top_result["label"]
    score = round(top_result["score"] * 100, 2)
    return f"Предсказание: {label} ({score}%)"

# Интерфейс Gradio
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите фото одежды"),
    outputs=gr.Textbox(label="Результат"),
    title="🛍️ Классификатор одежды",
    description="Загрузите изображение одежды, чтобы узнать, что это за вещь"
)

if __name__ == "__main__":
    app.launch()
