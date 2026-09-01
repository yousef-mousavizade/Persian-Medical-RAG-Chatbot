from pathlib import Path
import ollama

GENERATION_MODEL = "qwen2.5:3b-instruct"

SYSTEM_PROMPT = """تو یک دستیار اطلاعات دارویی هستی.

فقط بر اساس اطلاعات موجود در بخش «زمینه» (Context) پاسخ بده.
اگر پاسخ سؤال در زمینه وجود ندارد، اطلاعات را حدس نزن و صریحاً اعلام کن
که اطلاعات کافی در منابع موجود نیست.

پاسخ را به زبان فارسی، واضح و مختصر ارائه کن.

در مورد تشخیص قطعی بیماری یا تغییر خودسرانه‌ی دوز دارو، توصیه‌ی قطعی
ارائه نکن و کاربر را به مراجعه به پزشک یا داروساز ارجاع بده."""


def build_context(results) -> str:
    """Format retrieved (doc, score) pairs into a single context block."""
    parts = []
    for i, (doc, _score) in enumerate(results, start=1):
        drug = doc.metadata.get("drug_name", "نامشخص")
        # Strip the "passage: " prefix added during indexing - it's an
        # embedding-model requirement, not something the LLM should see.
        text = doc.page_content.removeprefix("passage: ")
        parts.append(f"[منبع {i} - دارو: {drug}]\n{text}")
    return "\n\n".join(parts)


def generate_answer(question: str, results) -> str:
    """Generate a Persian answer from the user's question and retrieved chunks.

    Requires the Ollama app to be running in the background (it starts
    automatically after installation on most systems).
    """
    context = build_context(results)
    prompt = f"زمینه (Context):\n{context}\n\nسؤال کاربر:\n{question}"

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},  # low temperature: grounded, consistent answers
    )

    return response["message"]["content"]
