import json
import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
DOCSTORE_PATH = os.path.join(ARTIFACTS_DIR, "docstore.pkl")

# Verify artifacts
for path, desc in [(CONFIG_PATH, "config.json"), (INDEX_PATH, "faiss.index"), (DOCSTORE_PATH, "docstore.pkl")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {desc} tại {path}")

# Load artifacts
with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
    config = json.load(f)

index = faiss.read_index(INDEX_PATH)
with open(DOCSTORE_PATH, "rb") as f:
    docstore = pickle.load(f)

EMB_MODEL = config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = config.get("llm_model", "Qwen/Qwen2-1.5B-Instruct")
DEFAULT_TOPK = int(config.get("top_k", 3))

# Hardware detection
is_cuda = torch.cuda.is_available()
device = "cuda" if is_cuda else "cpu"
dtype = torch.float16 if is_cuda else torch.float32

embedder = SentenceTransformer(EMB_MODEL, device=device)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=dtype,
    device_map="auto" if is_cuda else None,
    trust_remote_code=True
)
if not is_cuda:
    model.to("cpu")
model.eval()


def rag_answer(query: str, top_k: int = DEFAULT_TOPK, max_new_tokens: int = 384):
    """Truy hồi ngữ cảnh và sinh câu trả lời tư vấn pháp luật."""
    if not query.strip():
        return "Vui lòng nhập câu hỏi pháp luật.", []

    # 1. Retrieval
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, ids = index.search(q_vec, top_k)

    contexts, citations = [], []
    for i in ids[0]:
        if 0 <= i < len(docstore):
            item = docstore[i]
            if text := item.get("text", "").strip():
                contexts.append(text)
                citations.append(item.get("citation", "Nguồn không rõ").strip())

    if not contexts:
        return "Chưa đủ dữ liệu pháp luật để kết luận.", []

    context_str = "\n\n".join(f"[{idx + 1}] {ctx}" for idx, ctx in enumerate(contexts))

    # 2. Prompting
    system_prompt = (
        "Bạn là chatbot tư vấn pháp luật Việt Nam. "
        "Chỉ trả lời dựa trên ngữ cảnh pháp luật được cung cấp. "
        "Không suy đoán hoặc bịa thông tin. "
        "Nếu không đủ dữ liệu, hãy trả lời: 'Chưa đủ dữ liệu pháp luật để kết luận.'"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Câu hỏi: {query}\n\nNgữ cảnh pháp luật:\n{context_str}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # 3. Generation & Decoding
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    answer = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
    unique_citations = list(dict.fromkeys(citations))

    return answer, unique_citations
