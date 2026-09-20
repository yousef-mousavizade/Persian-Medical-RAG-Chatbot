"""
Adapted from the instructor's compare_eval.py template (fine-tuned
model vs prompt-engineering comparison) - repurposed for OUR project's
actual need: filling the "Abstention Accuracy: not formally measured"
gap identified in the evaluation report.

We have no fine-tuned classifier (out of scope for this project), so
instead we compare our two REAL candidate methods for deciding whether
a question is in-domain (answerable from our pharmaceutical dataset)
or out-of-domain (should trigger Abstention):

  Method A - "Threshold-based" (what production actually uses):
      run real retrieval and check the best FAISS similarity score
      against MIN_CONFIDENCE_SCORE.

  Method B - "Prompt-based": ask the LLM directly, in a single
      yes/no-style prompt, whether the question is about medication.

Requires: pip install ollama
          (plus everything retrieve.py already needs)
"""
import sys
import time
from pathlib import Path

import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from retrieval.retrieve import load_vectorstore, search  # noqa: E402

GENERATION_MODEL = "qwen2.5:7b-instruct"
MIN_CONFIDENCE_SCORE = 0.30  # keep in sync with src/api/main.py

# A small labeled test set: real in-domain drug questions used
# throughout this project's manual testing, plus clearly out-of-domain
# questions used earlier to test the Abstention path.
TEST_SAMPLES = [
    {"text": "آسپرین برای سردرد خوبه؟", "true_label": "in_domain"},
    {"text": "دوز مصرف سفیکسیم برای کودکان چقدره؟", "true_label": "in_domain"},
    {"text": "تفاوت فلوکستین و فلووکسامین چیه؟", "true_label": "in_domain"},
    {"text": "قرص فاموتیدین با چه داروهایی تداخل داره؟", "true_label": "in_domain"},
    {"text": "آیا شربت لاکتولوز برای یبوست کودکان مناسب است؟", "true_label": "in_domain"},
    {"text": "کلردیازپوکساید چه عوارض جانبی‌ای داره؟", "true_label": "in_domain"},
    {"text": "سیپروفلوکساسین برای عفونت ادراری چند روز باید مصرف بشه؟", "true_label": "in_domain"},
    {"text": "آیا موپیروسین برای زخم صورت هم استفاده میشه؟", "true_label": "in_domain"},
    {"text": "نظرت درباره داروهای گیاهی چینی چیه؟", "true_label": "out_of_domain"},
    {"text": "بهترین رژیم غذایی برای کاهش وزن چیه؟", "true_label": "out_of_domain"},
    {"text": "پایتخت فرانسه کجاست؟", "true_label": "out_of_domain"},
    {"text": "بهترین گوشی موبایل زیر ده میلیون تومان کدومه؟", "true_label": "out_of_domain"},
    {"text": "نظرت درباره وضعیت اقتصاد ایران چیه؟", "true_label": "out_of_domain"},
]


def evaluate_threshold_method(samples, vectorstore):
    print("Evaluating Method A: embedding confidence threshold...")
    results = []
    start = time.time()

    for sample in samples:
        query = "query: " + sample["text"]
        search_results = vectorstore.similarity_search_with_score(query, k=5)
        best_score = min(score for _, score in search_results) if search_results else float("inf")
        predicted = "in_domain" if best_score <= MIN_CONFIDENCE_SCORE else "out_of_domain"
        results.append({
            "text": sample["text"],
            "true": sample["true_label"],
            "predicted": predicted,
            "correct": predicted == sample["true_label"],
        })

    elapsed = time.time() - start
    return results, elapsed


def evaluate_prompt_method(samples):
    print(f"Evaluating Method B: prompt-based classification with {GENERATION_MODEL}...")
    results = []
    start = time.time()

    for sample in samples:
        prompt = (
            "آیا سؤال زیر دربارهٔ یک داروی خاص (مصرف، دوز، عوارض یا تداخل دارویی) است؟\n\n"
            f"سؤال: {sample['text']}\n\n"
            "فقط یکی از این دو کلمه را بنویس: بله یا خیر"
        )
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        answer_text = response["message"]["content"].strip()
        predicted = "in_domain" if "بله" in answer_text else "out_of_domain"
        results.append({
            "text": sample["text"],
            "true": sample["true_label"],
            "predicted": predicted,
            "correct": predicted == sample["true_label"],
        })

    elapsed = time.time() - start
    return results, elapsed


def print_comparison(a_results, a_time, b_results, b_time):
    print("\n" + "=" * 90)
    print(f"{'متن':<45} {'واقعی':<14} {'آستانه':<14} {'پرامپت':<14}")
    print("=" * 90)

    for a, b in zip(a_results, b_results):
        text = a["text"][:42]
        true = a["true"]
        a_pred = f"{'OK' if a['correct'] else 'ERR'} {a['predicted']}"
        b_pred = f"{'OK' if b['correct'] else 'ERR'} {b['predicted']}"
        print(f"{text:<45} {true:<14} {a_pred:<14} {b_pred:<14}")

    a_acc = sum(1 for r in a_results if r["correct"]) / len(a_results)
    b_acc = sum(1 for r in b_results if r["correct"]) / len(b_results)

    print("\n" + "=" * 90)
    print("مقایسه نهایی (Abstention Accuracy):")
    print("=" * 90)
    print("روش آستانهٔ Embedding (تولید فعلی):")
    print(f"  دقت: {a_acc:.1%}")
    print(f"  زمان کل: {a_time:.2f} ثانیه ({a_time/len(a_results):.2f}s / سؤال)")
    print()
    print(f"روش Prompt-based ({GENERATION_MODEL}):")
    print(f"  دقت: {b_acc:.1%}")
    print(f"  زمان کل: {b_time:.2f} ثانیه ({b_time/len(b_results):.2f}s / سؤال)")


def main():
    print("مقایسه دو روش تشخیص سؤال دارویی/غیردارویی (Abstention)")
    print("=" * 90)

    vectorstore = load_vectorstore()

    a_results, a_time = evaluate_threshold_method(TEST_SAMPLES, vectorstore)
    b_results, b_time = evaluate_prompt_method(TEST_SAMPLES)

    print_comparison(a_results, a_time, b_results, b_time)


if __name__ == "__main__":
    main()
