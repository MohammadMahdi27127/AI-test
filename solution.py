import argparse
import dataclasses
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any

@dataclasses.dataclass
class Model:
    name: str
    is_local: bool
    required_ram_gb: float
    context_window: int
    input_cost_per_1m: float  # دلار بر یک میلیون توکن
    output_cost_per_1m: float # دلار بر یک میلیون توکن
    supported_tasks: List[str]
    description: str

# لیست مدل‌های نمونه (قابل ویرایش و گسترش)
DEFAULT_MODELS = [
    Model(
        name="llama3:8b",
        is_local=True,
        required_ram_gb=8.0,
        context_window=8192,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        supported_tasks=["summarization", "qa", "content", "coding"],
        description="مدل محلی سبک، سریع و عالی برای سیستم‌های با رم متوسط."
    ),
    Model(
        name="deepseek-coder:6.7b",
        is_local=True,
        required_ram_gb=8.0,
        context_window=16384,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        supported_tasks=["coding", "summarization"],
        description="متخصص محلی در حوزه برنامه‌نویسی و تولید کد."
    ),
    Model(
        name="llama3:70b",
        is_local=True,
        required_ram_gb=40.0,
        context_window=8192,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        supported_tasks=["summarization", "qa", "content", "coding"],
        description="مدل محلی بسیار قدرتمند نیازمند سخت‌افزار حرفه‌ای."
    ),
    Model(
        name="gpt-4o-mini",
        is_local=False,
        required_ram_gb=0.0,
        context_window=128000,
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        supported_tasks=["summarization", "qa", "content", "coding"],
        description="مدل ابری اقتصادی، سریع و با پنجره زمینه بزرگ."
    ),
    Model(
        name="gpt-4o",
        is_local=False,
        required_ram_gb=0.0,
        context_window=128000,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        supported_tasks=["summarization", "qa", "content", "coding"],
        description="مدل ابری پرچم‌دار و بسیار دقیق برای پیچیده‌ترین کارها."
    )
]

def check_ollama_status() -> List[str]:
    """بررسی اتصال به Ollama و دریافت لیست مدل‌های محلی نصب‌شده"""
    url = "http://localhost:11434/api/tags"
    local_models = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                for model in data.get("models", []):
                    model_name = model.get("name", "")
                    if model_name:
                        local_models.append(model_name)
    except (urllib.error.URLError, TimeoutError, Exception):
        print("⚠️ [هشدار]: سرویس Ollama در دسترس نیست یا خاموش است. بررسی مدل‌های محلی صرفاً بر اساس لیست پیش‌فرض انجام می‌شود.")
    return local_models

def calculate_costs(requests: int, input_tokens: int, output_tokens: int, model: Model) -> float:
    """محاسبه هزینه ماهانه مدل ابری"""
    if model.is_local:
        return 0.0
    total_input = requests * input_tokens
    total_output = requests * output_tokens
    cost = (total_input / 1_000_000 * model.input_cost_per_1m) + (total_output / 1_000_000 * model.output_cost_per_1m)
    return cost

def evaluate_models(task: str, privacy: str, ram: float, requests: int, input_tokens: int, output_tokens: int, budget: float, models: List[Model], active_ollama_models: List[str]) -> List[Dict[str, Any]]:
    results = []
    total_input_tokens_req = input_tokens + output_tokens
    
    for model in models:
        score = 100
        pros = []
        cons = []
        is_disqualified = False
        disqualify_reason = ""

        # بررسی کارایی برای تسک مورد نظر
        if task not in model.supported_tasks:
            score -= 30
            cons.append("پشتیبانی ضعیف‌تر از نوع کارکرد انتخابی")

        # بررسی محدودیت RAM برای مدل‌های محلی
        if model.is_local:
            if ram < model.required_ram_gb:
                is_disqualified = True
                disqualify_reason = f"رم ناکافی سیستم (نیاز به {model.required_ram_gb} گیگابایت)"
            else:
                score += 15
                pros.append("اجرای کاملاً محلی و رایگان")
                if active_ollama_models and any(model.name in om for om in active_ollama_models):
                    score += 10
                    pros.append("مدل روی سرویس Ollama شما نصب و آماده به کار است")

            # اعمال امتیاز حریم خصوصی
            if privacy == "high":
                score += 40
                pros.append("حریم خصوصی ۱۰۰ درصد (مناسب سطح امنیتی بالا)")
            elif privacy == "medium":
                score += 20
        else:
            # مدل‌های ابری
            cost = calculate_costs(requests, input_tokens, output_tokens, model)
            
            if privacy == "high":
                score -= 50
                cons.append("ارسال داده‌ها به بستر ابری (مغایر با سطح حریم خصوصی بالا)")
            
            if cost > budget:
                score -= 40
                cons.append(f"هزینه ماهانه (${cost:.2f}) بالاتر از بودجه مصوب (${budget}) است")
            else:
                pros.append(f"هزینه تخمینی مقرون‌به‌صرفه: ${cost:.2f}")

            pros.append(f"پنجره زمینه بسیار بزرگ ({model.context_window} توکن)")

        # بررسی Context Window
        if total_input_tokens_req > model.context_window:
            is_disqualified = True
            disqualify_reason = f"حجم توکن درخواست از محدودیت context window مدل بیشتر است ({model.context_window})"

        if is_disqualified:
            score = 0
            cons.append(f"رد صلاحیت: {disqualify_reason}")

        results.append({
            "model": model,
            "score": max(0, score),
            "pros": pros,
            "cons": cons,
            "estimated_cost": calculate_costs(requests, input_tokens, output_tokens, model)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def main():
    parser = argparse.ArgumentParser(description="ابزار انتخاب و رتبه‌بندی مدل‌های هوش مصنوعی محلی و ابری")
    parser.add_argument("--task", type=str, help="نوع کاربری")
    parser.add_argument("--privacy", type=str, help="سطح حریم خصوصی مورد نیاز")
    parser.add_argument("--ram", type=float, help="میزان RAM در دسترس سیستم (گیگابایت)")
    parser.add_argument("--requests", type=int, help="تعداد درخواست ماهانه")
    parser.add_argument("--input-tokens", type=int, help="میانگین توکن ورودی")
    parser.add_argument("--output-tokens", type=int, help="میانگین توکن خروجی")
    parser.add_argument("--budget", type=float, help="بودجه ماهانه (دلار)")
    parser.add_argument("--check-ollama", action="store_true", help="بررسی وضعیت الاما و مدل‌های لوکال")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("✨ سامانه ارزیابی و انتخاب هوشمند مدل‌های هوش مصنوعی")
    print("="*60)

    # دریافت مقادیر به صورت تعاملی اگر از طریق کامندلاین وارد نشده باشند
    task = args.task
    if not task:
        print("\nلطفاً نوع کاربری مورد نظر را انتخاب کنید:")
        print("  1) qa (پرسش و پاسخ)")
        print("  2) summarization (خلاصه‌سازی)")
        print("  3) content (تولید محتوا)")
        print("  4) coding (کدنویسی)")
        choice = input("انتخاب شما [پیش‌فرض: qa]: ").strip().lower()
        mapping = {"1": "qa", "2": "summarization", "3": "content", "4": "coding"}
        task = mapping.get(choice, choice if choice in ["qa", "summarization", "content", "coding"] else "qa")

    privacy = args.privacy
    if not privacy:
        print("\nسطح حریم خصوصی مورد نیاز را مشخص کنید:")
        print("  1) low (پایین)")
        print("  2) medium (متوسط)")
        print("  3) high (بالا)")
        choice = input("انتخاب شما [پیش‌فرض: medium]: ").strip().lower()
        mapping = {"1": "low", "2": "medium", "3": "high"}
        privacy = mapping.get(choice, choice if choice in ["low", "medium", "high"] else "medium")

    ram = args.ram
    if ram is None:
        ram_input = input("\nمیزان RAM در دسترس سیستم به گیگابایت [پیش‌فرض: 16]: ").strip()
        ram = float(ram_input) if ram_input else 16.0

    requests = args.requests
    if requests is None:
        req_input = input("\nتعداد درخواست ماهانه [پیش‌فرض: 5000]: ").strip()
        requests = int(req_input) if req_input else 5000

    input_tokens = args.input_tokens
    if input_tokens is None:
        in_tok_input = input("\nمیانگین توکن ورودی در هر درخواست [پیش‌فرض: 1000]: ").strip()
        input_tokens = int(in_tok_input) if in_tok_input else 1000

    output_tokens = args.output_tokens
    if output_tokens is None:
        out_tok_input = input("\nمیانگین توکن خروجی در هر درخواست [پیش‌فرض: 300]: ").strip()
        output_tokens = int(out_tok_input) if out_tok_input else 300

    budget = args.budget
    if budget is None:
        bud_input = input("\nبودجه ماهانه به دلار [پیش‌فرض: 50.0]: ").strip()
        budget = float(bud_input) if bud_input else 50.0

    check_ollama = args.check_ollama
    if not check_ollama:
        check_input = input("\nآیا می‌خواهید وضعیت اتصال به Ollama بررسی شود؟ (y/n) [پیش‌فرض: n]: ").strip().lower()
        if check_input == 'y':
            check_ollama = True

    print("\n" + "-"*60)
    active_ollama_models = []
    if check_ollama:
        print("🔍 در حال بررسی اتصال به Ollama...")
        active_ollama_models = check_ollama_status()
        if active_ollama_models:
            print(f"✅ مدل‌های فعال یافت شده در Ollama: {', '.join(active_ollama_models)}")

    monthly_total_tokens = requests * (input_tokens + output_tokens)
    print(f"\n📊 اطلاعات کلی پردازش:")
    print(f"   • نوع کار: {task} | حریم خصوصی: {privacy} | رم سیستم: {ram}GB")
    print(f"   • حجم کل توکن‌های ماهانه: {monthly_total_tokens:,} توکن")
    print(f"   • بودجه ماهانه تعیین شده: ${budget}")
    print("-" * 60)

    # ارزیابی مدل‌ها
    evaluations = evaluate_models(task, privacy, ram, requests, input_tokens, output_tokens, budget, DEFAULT_MODELS, active_ollama_models)

    print("\n🏆 ۳ مدل برتر پیشنهادی:")
    for i, item in enumerate(evaluations[:3], 1):
        model = item["model"]
        m_type = "محلی (Local)" if model.is_local else "ابری (Cloud)"
        print(f"\n{i}. نام مدل: **{model.name}** ({m_type})")
        print(f"   ⭐ امتیاز نهایی: {item['score']} / 100")
        print(f"   💰 هزینه ماهانه تخمینی: ${item['estimated_cost']:.2f}")
        print(f"   📝 توضیحات: {model.description}")
        
        if item["pros"]:
            print("   🟢 مزایا:")
            for pro in item["pros"]:
                print(f"      - {pro}")
        if item["cons"]:
            print("   🔴 محدودیت‌ها / معایب:")
            for con in item["cons"]:
                print(f"      - {con}")

    print("\n" + "="*60)
    print("پروژه با موفقیت تحلیل و اجرا شد مهدی عزیز! 🚀\n")

if __name__ == "__main__":
    main()