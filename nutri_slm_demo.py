
import subprocess, sys

# =============================================================================
# SECTION 1 — Install dependencies
# =============================================================================

def install():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "bitsandbytes>=0.43.0", "--prefer-binary",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "triton",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers==4.51.3", "peft==0.14.0", "trl==0.13.0",
        "accelerate==1.3.0", "datasets==3.2.0",
        "gradio==5.9.1", "sentence-transformers==3.3.1",
        "faiss-cpu==1.9.0", "mlflow==2.19.0",
    ])
    import bitsandbytes as bnb
    if not hasattr(bnb.nn, "Linear4bit"):
        print("WARNING: bitsandbytes CUDA build not detected.")
    else:
        print("All packages installed — bitsandbytes CUDA build OK")

install()


# =============================================================================
# SECTION 2 — North India food database (IFCT 2017, public domain)
# =============================================================================

import json


FOOD_DB = [
    # ── IRON ──────────────────────────────────────────────────────────────────
    {
        "name": "Palak (Spinach)", "hinglish_name": "palak",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Iron", "Vitamin C", "Calcium"],
        "season": ["winter"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 katori cooked (100g)",
        "nutrients": {"Iron_mg": 2.7, "Vitamin_C_mg": 28, "Calcium_mg": 73},
    },
    {
        "name": "Rajma (Kidney Beans)", "hinglish_name": "rajma",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Iron", "Protein", "Zinc"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 katori cooked (100g)",
        "nutrients": {"Iron_mg": 2.9, "Protein_g": 8.7, "Zinc_mg": 1.2},
    },
    {
        "name": "Kala Chana", "hinglish_name": "kala chana",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Iron", "Protein", "Zinc"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 katori cooked (100g)",
        "nutrients": {"Iron_mg": 3.2, "Protein_g": 7.2, "Zinc_mg": 1.0},
    },
    {
        "name": "Chicken Liver", "hinglish_name": "murge ka kaleji",
        "diet_types": ["non-vegetarian"],
        "deficiencies": ["Iron", "Vitamin B12", "Vitamin A"],
        "season": ["all"], "region_tags": ["Delhi", "UP", "Haryana"],
        "portion": "75g cooked",
        "nutrients": {"Iron_mg": 9.0, "Vitamin_B12_mcg": 16.5, "Vitamin_A_mcg": 4968},
    },
    {
        "name": "Methi (Fenugreek Leaves)", "hinglish_name": "methi",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Iron", "Calcium"],
        "season": ["winter"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 katori cooked (100g)",
        "nutrients": {"Iron_mg": 1.9, "Calcium_mg": 395},
    },
    # ── VITAMIN D ─────────────────────────────────────────────────────────────
    {
        "name": "Mushroom (sun-exposed)", "hinglish_name": "khumb",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Vitamin D"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "100g cooked",
        "nutrients": {"Vitamin_D_IU": 400},
    },
    {
        "name": "Egg (whole)", "hinglish_name": "anda",
        "diet_types": ["non-vegetarian"],
        "deficiencies": ["Vitamin D", "Vitamin B12", "Protein"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "2 whole eggs",
        "nutrients": {"Vitamin_D_IU": 82, "Vitamin_B12_mcg": 1.1, "Protein_g": 12.6},
    },
    {
        "name": "Fortified Milk (toned)", "hinglish_name": "toned doodh",
        "diet_types": ["vegetarian", "jain"],
        "deficiencies": ["Vitamin D", "Calcium", "Vitamin B12"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 glass (250ml)",
        "nutrients": {"Vitamin_D_IU": 120, "Calcium_mg": 285, "Vitamin_B12_mcg": 0.9},
    },
    {
        "name": "Rohu Fish", "hinglish_name": "rohu machli",
        "diet_types": ["non-vegetarian"],
        "deficiencies": ["Vitamin D", "Omega-3", "Protein"],
        "season": ["all"], "region_tags": ["UP", "Delhi"],
        "portion": "100g cooked",
        "nutrients": {"Vitamin_D_IU": 360, "Omega3_g": 0.4, "Protein_g": 19},
    },
    # ── VITAMIN B12 ───────────────────────────────────────────────────────────
    {
        "name": "Dahi (Curd)", "hinglish_name": "dahi",
        "diet_types": ["vegetarian", "jain"],
        "deficiencies": ["Vitamin B12", "Calcium"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 katori (150g)",
        "nutrients": {"Vitamin_B12_mcg": 0.6, "Calcium_mg": 180},
    },
    {
        "name": "Paneer", "hinglish_name": "paneer",
        "diet_types": ["vegetarian", "jain"],
        "deficiencies": ["Vitamin B12", "Calcium", "Protein"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "100g",
        "nutrients": {"Vitamin_B12_mcg": 0.8, "Calcium_mg": 190, "Protein_g": 18},
    },
    {
        "name": "Nutritional Yeast", "hinglish_name": "nutritional yeast",
        "diet_types": ["vegan"],
        "deficiencies": ["Vitamin B12"],
        "season": ["all"], "region_tags": ["Delhi"],
        "portion": "2 tbsp (15g)",
        "nutrients": {"Vitamin_B12_mcg": 4.0},
    },
    # ── CALCIUM ───────────────────────────────────────────────────────────────
    {
        "name": "Ragi (Finger Millet)", "hinglish_name": "ragi / nachni",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Calcium", "Iron"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 roti (30g flour)",
        "nutrients": {"Calcium_mg": 105, "Iron_mg": 1.1},
    },
    {
        "name": "Til (Sesame Seeds)", "hinglish_name": "til",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Calcium", "Iron", "Zinc"],
        "season": ["winter"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "2 tbsp (20g)",
        "nutrients": {"Calcium_mg": 220, "Iron_mg": 2.4, "Zinc_mg": 1.4},
    },
    {
        "name": "Lassi", "hinglish_name": "lassi",
        "diet_types": ["vegetarian", "jain"],
        "deficiencies": ["Calcium", "Vitamin B12"],
        "season": ["summer"], "region_tags": ["Punjab", "Delhi", "Haryana"],
        "portion": "1 glass (250ml)",
        "nutrients": {"Calcium_mg": 300, "Vitamin_B12_mcg": 0.7},
    },
    # ── ZINC ──────────────────────────────────────────────────────────────────
    {
        "name": "Kaddu ke Beej (Pumpkin Seeds)", "hinglish_name": "kaddu ke beej",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Zinc", "Omega-3", "Iron"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "2 tbsp (20g)",
        "nutrients": {"Zinc_mg": 1.8, "Iron_mg": 1.6},
    },
    {
        "name": "Mutton (lean)", "hinglish_name": "gosht",
        "diet_types": ["non-vegetarian"],
        "deficiencies": ["Zinc", "Iron", "Vitamin B12"],
        "season": ["all"], "region_tags": ["Delhi", "UP", "Haryana"],
        "portion": "100g cooked",
        "nutrients": {"Zinc_mg": 4.8, "Iron_mg": 2.5, "Vitamin_B12_mcg": 2.1},
    },
    # ── VITAMIN C ─────────────────────────────────────────────────────────────
    {
        "name": "Amla (Indian Gooseberry)", "hinglish_name": "amla",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Vitamin C"],
        "season": ["winter"], "region_tags": ["UP", "Delhi", "Haryana"],
        "portion": "2 amla (60g)",
        "nutrients": {"Vitamin_C_mg": 420},
    },
    {
        "name": "Guava", "hinglish_name": "amrood",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Vitamin C"],
        "season": ["winter"], "region_tags": ["UP", "Delhi", "Haryana", "Punjab"],
        "portion": "1 medium (100g)",
        "nutrients": {"Vitamin_C_mg": 228},
    },
    # ── OMEGA-3 ───────────────────────────────────────────────────────────────
    {
        "name": "Alsi (Flaxseeds)", "hinglish_name": "alsi ke beej",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Omega-3"],
        "season": ["all"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "1 tbsp ground (10g)",
        "nutrients": {"Omega3_g": 2.3},
    },
    {
        "name": "Akhrot (Walnuts)", "hinglish_name": "akhrot",
        "diet_types": ["vegetarian", "vegan", "jain"],
        "deficiencies": ["Omega-3"],
        "season": ["winter"], "region_tags": ["Punjab", "Delhi", "UP", "Haryana"],
        "portion": "4 halves (30g)",
        "nutrients": {"Omega3_g": 2.6},
    },
]

print(f"Food database loaded: {len(FOOD_DB)} foods")
print(f"  Deficiencies : {sorted(set(d for f in FOOD_DB for d in f['deficiencies']))}")
print(f"  Diet types   : {sorted(set(d for f in FOOD_DB for d in f['diet_types']))}")


# =============================================================================
# SECTION 3 — Synthetic training data generator (no API needed)
# =============================================================================

import random
from datasets import Dataset

DEFICIENCIES = ["Iron", "Vitamin D", "Vitamin B12", "Calcium",
                "Zinc", "Vitamin C", "Omega-3"]
DIET_TYPES   = ["vegetarian", "non-vegetarian", "vegan", "jain"]
REGIONS      = ["Delhi", "Punjab", "UP", "Haryana"]
SEASONS      = ["summer", "winter", "monsoon"]
AGE_GROUPS   = [
    (18, 25, "young adult"),
    (26, 40, "adult"),
    (41, 60, "middle-aged"),
    (61, 80, "senior"),
]

HINGLISH_TEMPLATES = [
    "{deficiency} ki kami hai mujhe. Main {region} mein rehta/rehti hoon, {diet} hoon. Kya khaaon?",
    "{deficiency} deficiency hai meri. {diet} hoon, {region} se hoon. Weekly plan do.",
    "Doctor ne bola {deficiency} low hai. Main {age_group} hoon, {diet} khata/khati hoon. Suggest karo.",
    "Meri {deficiency} bahut kam hai. {region} mein {season} chal raha hai. {diet} hoon, kya khaaon?",
]

ENGLISH_TEMPLATES = [
    "I have {deficiency} deficiency. I'm a {diet} from {region}. What should I eat?",
    "My doctor found low {deficiency}. I follow a {diet} diet. Give me a weekly meal plan.",
    "I'm a {age_group} {diet} living in {region}. I need help with {deficiency} deficiency.",
    "It's {season} in {region}. I have {deficiency} deficiency and I'm {diet}. Suggest foods.",
]

TIPS = {
    "Iron":        [
        "Have lemon/amla with iron-rich meals to boost absorption.",
        "Avoid chai or coffee right after iron-rich meals.",
        "Cook in an iron kadai — it adds dietary iron naturally.",
    ],
    "Vitamin D":   [
        "Get 15-20 min of morning sunlight (before 10am) daily.",
        "Mushrooms left in sunlight for 30 min triple their Vitamin D.",
        "Vitamin D needs Vitamin K2 — add til and green veggies.",
    ],
    "Vitamin B12": [
        "B12 is found almost exclusively in animal/fermented foods.",
        "If vegan, a B12 supplement is strongly recommended.",
        "Fermented foods like idli/dosa batter have small B12 amounts.",
    ],
    "Calcium":     [
        "Pair calcium foods with Vitamin D for better absorption.",
        "Ragi roti is one of the best calcium sources in Indian diet.",
        "Avoid very high spinach with calcium foods — oxalates block it.",
    ],
    "Zinc":        [
        "Soak lentils/legumes overnight to reduce phytates that block zinc.",
        "Non-veg sources of zinc are more bioavailable than plant sources.",
    ],
    "Vitamin C":   [
        "Eat Vitamin C rich foods fresh — cooking destroys it.",
        "2 amla = 7x the Vitamin C of an orange.",
    ],
    "Omega-3":     [
        "Grind alsi before eating — whole seeds pass undigested.",
        "Store ground flaxseeds in the fridge to prevent rancidity.",
    ],
}


def get_foods_for(deficiency, diet_type, season, n=4):
    """Filter FOOD_DB by deficiency, diet type and season."""
    candidates = [
        f for f in FOOD_DB
        if deficiency in f["deficiencies"]
        and diet_type in f["diet_types"]
        and (season in f["season"] or "all" in f["season"])
    ]
    if not candidates:                      # fallback: ignore season
        candidates = [
            f for f in FOOD_DB
            if deficiency in f["deficiencies"]
            and diet_type in f["diet_types"]
        ]
    return random.sample(candidates, min(n, len(candidates)))


def make_goal_plan(deficiency, diet_type, season, foods):
    """Build structured goal plan. Safe for 1+ foods (pads to 3)."""
    while len(foods) < 3:
        foods = foods + foods
    foods      = foods[:3]
    food_names = [f["name"] for f in foods]

    daily = {
        "morning":   f"{foods[0]['name']} ({foods[0]['portion']})",
        "afternoon": f"{foods[1]['name']} ({foods[1]['portion']})",
        "evening":   f"{foods[2]['name']} ({foods[2]['portion']})",
        "tip":       random.choice(TIPS.get(deficiency, ["Eat diverse, whole foods."])),
    }
    weekly = (
        f"Include {food_names[0]} at least 4x this week. "
        f"Try {food_names[1]} every alternate day."
    )
    if food_names[2] != food_names[0]:
        weekly += f" Add {food_names[2]} as a snack 3x a week."
    monthly = (
        f"After 4 weeks of this plan, retest {deficiency} levels. "
        f"Target: reach normal range."
    )
    return {"daily_plan": daily, "weekly_goal": weekly, "monthly_milestone": monthly}


def generate_example(lang="english"):
    """Generate one instruction-tuning example."""
    deficiency            = random.choice(DEFICIENCIES)
    diet_type             = random.choice(DIET_TYPES)
    region                = random.choice(REGIONS)
    season                = random.choice(SEASONS)
    age_min, age_max, age_label = random.choice(AGE_GROUPS)
    age                   = random.randint(age_min, age_max)

    templates   = HINGLISH_TEMPLATES if lang == "hinglish" else ENGLISH_TEMPLATES
    instruction = random.choice(templates).format(
        deficiency=deficiency, diet=diet_type, region=region,
        season=season, age=age, age_group=age_label,
    )

    foods = get_foods_for(deficiency, diet_type, season)
    if not foods:
        return None

    goal_plan = make_goal_plan(deficiency, diet_type, season, foods)

    output = {
        "deficiency":        deficiency,
        "diet_type":         diet_type,
        "region":            region,
        "top_foods": [
            {
                "name":         f["name"],
                "hinglish_name": f["hinglish_name"],
                "portion":      f["portion"],
                "nutrients":    f["nutrients"],
            }
            for f in foods
        ],
        "daily_plan":        goal_plan["daily_plan"],
        "weekly_goal":       goal_plan["weekly_goal"],
        "monthly_milestone": goal_plan["monthly_milestone"],
    }

    prompt = (
        "<|im_start|>system\n"
        "You are NutriSLM, a nutrition advisor specialized in North Indian diets. "
        "Suggest region-appropriate foods for vitamin/mineral deficiencies and create "
        "daily, weekly, and monthly goals. Always respond in structured JSON.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{json.dumps(output, ensure_ascii=False, indent=2)}<|im_end|>"
    )
    return {"text": prompt, "deficiency": deficiency,
            "diet_type": diet_type, "lang": lang}


# Generate 6 000 examples (4 k English + 2 k Hinglish)
random.seed(42)
examples = []
for _ in range(4000):
    ex = generate_example("english")
    if ex:
        examples.append(ex)
for _ in range(2000):
    ex = generate_example("hinglish")
    if ex:
        examples.append(ex)

random.shuffle(examples)
dataset  = Dataset.from_list(examples)
split    = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"Dataset ready — train: {len(train_ds)}, eval: {len(eval_ds)}")
print("\nSample prompt (first 500 chars):")
print(train_ds[0]["text"][:500], "...")


# =============================================================================
# SECTION 4 — Load Qwen3-0.6B with QLoRA (fits on free T4)
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "Qwen/Qwen3-0.6B"

print("Loading tokenizer...")
tokenizer              = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in 4-bit (QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

print("Attaching LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =============================================================================
# SECTION 5 — Fine-tune with SFTTrainer (~35-45 min on T4)
# =============================================================================

from trl import SFTTrainer, SFTConfig
import mlflow

mlflow.set_experiment("NutriSLM")

sft_config = SFTConfig(
    output_dir="./nutri_slm_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    max_seq_length=1024,
    dataset_text_field="text",
    report_to="none",                    # set to "mlflow" to enable logging
    dataloader_num_workers=2,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

print("Starting fine-tuning...")
with mlflow.start_run(run_name="qwen3-0.6b-nutri-v1"):
    mlflow.log_params({
        "model": MODEL_ID, "lora_r": 16, "lora_alpha": 32,
        "lr": 2e-4, "epochs": 3, "train_size": len(train_ds),
    })
    trainer.train()
    mlflow.log_metric(
        "final_train_loss",
        trainer.state.log_history[-1].get("train_loss", 0),
    )

print("Fine-tuning complete!")
model.save_pretrained("./nutri_slm_adapter")
tokenizer.save_pretrained("./nutri_slm_adapter")
print("Adapter saved to ./nutri_slm_adapter")


# =============================================================================
# SECTION 6 — Inference helper
# =============================================================================

def build_prompt(user_input: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are NutriSLM, a nutrition advisor specialized in North Indian diets. "
        "Suggest region-appropriate foods for vitamin/mineral deficiencies and create "
        "daily, weekly, and monthly goals. Always respond in structured JSON.<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_inference(user_input: str, max_new_tokens: int = 600) -> dict:
    prompt  = build_prompt(user_input)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded        = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assistant_part = decoded.split("<|im_start|>assistant")[-1].strip()
    try:
        start = assistant_part.find("{")
        end   = assistant_part.rfind("}") + 1
        return json.loads(assistant_part[start:end])
    except Exception:
        return {"raw_output": assistant_part}


# Quick smoke test
print("\n--- Inference test ---")
result = run_inference(
    "I have Iron deficiency. I'm vegetarian, from Delhi. What should I eat?"
)
print(json.dumps(result, indent=2, ensure_ascii=False))


# =============================================================================
# SECTION 7 — SQLite daily tracker
# =============================================================================

import sqlite3
from datetime import datetime

DB_PATH = "nutri_tracker.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   TEXT PRIMARY KEY,
            name      TEXT,
            region    TEXT,
            diet_type TEXT,
            created   TEXT
        );
        CREATE TABLE IF NOT EXISTS food_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT,
            food_name TEXT,
            meal_time TEXT,
            log_date  TEXT
        );
        CREATE TABLE IF NOT EXISTS goals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      TEXT,
            goal_type    TEXT,
            goal_text    TEXT,
            target_date  TEXT,
            completed    INTEGER DEFAULT 0,
            created_date TEXT
        );
    """)
    conn.commit()
    conn.close()


def upsert_user(user_id, name, region, diet_type):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO users VALUES (?,?,?,?,?)",
        (user_id, name, region, diet_type, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def log_food(user_id, food_name, meal_time="lunch"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO food_logs (user_id, food_name, meal_time, log_date) VALUES (?,?,?,?)",
        (user_id, food_name, meal_time, datetime.now().date().isoformat()),
    )
    conn.commit()
    conn.close()


def save_goals(user_id, plan: dict):
    conn  = sqlite3.connect(DB_PATH)
    today = datetime.now().date().isoformat()
    for gtype, gtext in [
        ("daily",   str(plan.get("daily_plan", ""))),
        ("weekly",  plan.get("weekly_goal", "")),
        ("monthly", plan.get("monthly_milestone", "")),
    ]:
        conn.execute(
            "INSERT INTO goals "
            "(user_id, goal_type, goal_text, target_date, created_date) "
            "VALUES (?,?,?,?,?)",
            (user_id, gtype, gtext, today, today),
        )
    conn.commit()
    conn.close()


def get_weekly_summary(user_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT food_name, meal_time, log_date "
        "FROM food_logs WHERE user_id=? "
        "ORDER BY log_date DESC LIMIT 50",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


init_db()
print("SQLite tracker DB initialised")


# =============================================================================
# SECTION 8 — Gradio app (4 tabs)
# =============================================================================

import gradio as gr
import uuid

SESSION = {}  # in-memory session store (per process)

DEFICIENCY_LIST = ["Iron", "Vitamin D", "Vitamin B12", "Calcium",
                   "Zinc", "Vitamin C", "Omega-3"]
DIET_LIST       = ["Vegetarian", "Non-vegetarian", "Vegan", "Jain"]
REGION_LIST     = ["Delhi", "Punjab", "UP", "Haryana"]


def setup_profile(name, region, diet):
    uid = str(uuid.uuid4())[:8]
    SESSION.update({
        "user_id": uid, "name": name,
        "region": region, "diet_type": diet.lower(),
    })
    upsert_user(uid, name, region, diet.lower())
    return f"Profile saved! Welcome, {name}. User ID: {uid}"


def get_plan(deficiency, extra_notes):
    if "user_id" not in SESSION:
        return "Please set up your profile first (Tab 1).", ""
    user_input = (
        f"My name is {SESSION['name']}. I have {deficiency} deficiency. "
        f"I am {SESSION['diet_type']}, from {SESSION['region']}. "
        f"{extra_notes} "
        f"Give me food suggestions and a daily, weekly, and monthly goal plan."
    )
    result = run_inference(user_input)
    save_goals(SESSION["user_id"], result)

    summary = f"## Plan for {deficiency} deficiency\n\n"
    if "top_foods" in result:
        summary += "### Top foods\n"
        for f in result["top_foods"]:
            summary += f"- **{f['name']}** — {f.get('portion', '')}\n"
    if "daily_plan" in result:
        summary += "\n### Daily plan\n"
        for k, v in result["daily_plan"].items():
            summary += f"- **{k.title()}**: {v}\n"
    if "weekly_goal" in result:
        summary += f"\n### Weekly goal\n{result['weekly_goal']}\n"
    if "monthly_milestone" in result:
        summary += f"\n### Monthly milestone\n{result['monthly_milestone']}\n"

    return summary, json.dumps(result, indent=2, ensure_ascii=False)


def log_meal(food_name, meal_time):
    if "user_id" not in SESSION:
        return "Please set up your profile first."
    log_food(SESSION["user_id"], food_name, meal_time)
    return f"Logged: {food_name} at {meal_time}"


def show_summary():
    if "user_id" not in SESSION:
        return "Please set up your profile first."
    rows = get_weekly_summary(SESSION["user_id"])
    if not rows:
        return "No food logs yet. Start logging in Tab 3!"
    out = "### Recent food logs\n\n| Food | Meal | Date |\n|------|------|------|\n"
    for food, meal, date in rows:
        out += f"| {food} | {meal} | {date} |\n"
    return out


with gr.Blocks(title="NutriSLM", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# NutriSLM — North India Nutrition & Deficiency Advisor")
    gr.Markdown("*Qwen3-0.6B fine-tuned · 100% open-source · North India regional foods*")

    with gr.Tab("1. Setup profile"):
        gr.Markdown("### Tell us about yourself")
        name_in     = gr.Textbox(label="Your name", placeholder="e.g. Priya")
        region_in   = gr.Dropdown(REGION_LIST, label="Your region", value="Delhi")
        diet_in     = gr.Dropdown(DIET_LIST,   label="Diet type",   value="Vegetarian")
        save_btn    = gr.Button("Save profile", variant="primary")
        profile_out = gr.Textbox(label="Status", interactive=False)
        save_btn.click(setup_profile, [name_in, region_in, diet_in], profile_out)

    with gr.Tab("2. Get nutrition plan"):
        gr.Markdown("### Enter your deficiency — get food suggestions + goals")
        deficiency_in = gr.Dropdown(DEFICIENCY_LIST, label="Deficiency", value="Iron")
        notes_in = gr.Textbox(
            label="Extra notes (optional — English or Hinglish)",
            placeholder="e.g. Mujhe bahut thakaan rehti hai / I feel very tired",
        )
        plan_btn = gr.Button("Generate my plan", variant="primary")
        plan_out = gr.Markdown(label="Your plan")
        json_out = gr.Code(label="Raw JSON", language="json")
        plan_btn.click(get_plan, [deficiency_in, notes_in], [plan_out, json_out])

    with gr.Tab("3. Log today's food"):
        gr.Markdown("### Log what you ate")
        food_in = gr.Textbox(label="Food name", placeholder="e.g. Rajma chawal")
        meal_in = gr.Dropdown(
            ["morning", "lunch", "evening", "dinner"],
            label="Meal time", value="lunch",
        )
        log_btn = gr.Button("Log food", variant="primary")
        log_out = gr.Textbox(label="Status", interactive=False)
        log_btn.click(log_meal, [food_in, meal_in], log_out)

    with gr.Tab("4. Weekly summary"):
        gr.Markdown("### Your food log history")
        refresh_btn = gr.Button("Load my summary", variant="primary")
        summary_out = gr.Markdown()
        refresh_btn.click(show_summary, [], summary_out)

demo.launch(share=True)

