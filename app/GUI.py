import gradio as gr
from rag_engine import rag_answer


def chat_fn(question: str):
    answer, citations = rag_answer(question)
    cite_text = "\n".join(f"• {c}" for c in citations) if citations else ""
    return answer, cite_text


with gr.Blocks(title="Chatbot Luật Việt Nam") as demo:
    gr.Markdown("# 🏛️ Chatbot Tư Vấn Luật Việt Nam (RAG)")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Câu hỏi pháp luật",
                placeholder="Nhập câu hỏi tại đây... (VD: Điều kiện đăng ký kết hôn là gì?)",
                lines=3,
            )
            btn = gr.Button("Hỏi Chatbot", variant="primary")
        with gr.Column():
            answer = gr.Textbox(label="Câu trả lời", lines=8)
            citations = gr.Textbox(label="Nguồn trích dẫn", lines=4)

    btn.click(fn=chat_fn, inputs=question, outputs=[answer, citations])
    question.submit(fn=chat_fn, inputs=question, outputs=[answer, citations])

if __name__ == "__main__":
    demo.launch()
