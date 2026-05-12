import pickle
import gradio as gr

# Model ve vectorizer yükle
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_news(text):
    if text.strip() == "":
        return "Lütfen bir haber metni giriniz."

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    if prediction == 0:
        return "🟥 Sahte Haber"
    else:
        return "🟩 Gerçek Haber"

demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Haber metnini buraya yazınız..."
    ),
    outputs="text",
    title="Fake News Detection System",
    description="TF-IDF ve makine öğrenmesi modelleri ile gerçek/sahte haber tespiti"
)

if __name__ == "__main__":
    demo.launch()