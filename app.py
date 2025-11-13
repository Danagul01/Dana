import gradio as gr
from transformers import pipeline
import torch

# Ограничиваем использование CPU, чтобы не съедать всю память
torch.set_num_threads(1)

# Инициализируем модель только при первом запросе
classifier = None

def classify_text(text):
    global classifier
    if classifier is None:
        # Маленькая модель, экономит память
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    result = classifier(text)[0]
    return f"Label: {result['label']}, Score: {result['score']:.2f}"

# Gradio интерфейс
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=2, placeholder="Введите текст..."),
    outputs="text",
    title="Text Classifier",
    description="Лёгкая версия модели для Render Free Tier"
)

if __name__ == "__main__":
    # Важно для Render Free Tier
    demo.launch(server_name="0.0.0.0", server_port=10000)
