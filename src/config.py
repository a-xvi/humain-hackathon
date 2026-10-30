"""All settings in one place. Change a value here, not inside the code."""

# --- LLMs (Gemini) ---
TEACHER_MODEL = "gemini-3.6-flash"         # labels messages; pick the fastest model that works (notebook step 2)
GENERATOR_MODEL = "gemini-3.5-flash-lite"  # writes synthetic messages; a DIFFERENT model than the teacher (less bias)
MIN_SECONDS_BETWEEN_CALLS = 4.0            # free-tier pacing (~15 requests/min)
TEACHER_BATCH_SIZE = 20                    # messages labeled per API call

# --- Data sizes ---
POOL_PER_CALL = 20        # messages per generation call
POOL_CALLS_PER_CELL = 3   # calls per (department x dialect) -> 8 x 4 x 3 x 20 = 1,920 pool messages
TEST_PER_CELL = 10        # 8 x 4 x 10 = 320 test messages
STRESS_PER_LABEL = 10     # per stress-set type and department

# --- Student ---
STUDENT_MODEL = "UBC-NLP/MARBERTv2"   # Arabic BERT pretrained on 1B Arabic tweets (dialects incl. Saudi)
MAX_LEN = 64
EPOCHS = 4
LR = 3e-5
BATCH_SIZE = 16

# --- Active distillation ---
SEED_SIZE = 80       # first random batch the teacher labels
STEP_SIZE = 80       # labels requested per round
BUDGET = 640         # max teacher labels per run
SEEDS = [0, 1, 2]    # repeat every experiment with 3 seeds
STRATEGIES = ["random", "margin", "margin_diverse"]

# --- Paths ---
DATA = "data"
EXP = "experiments"
MODEL_DIR = "models/student_final"
