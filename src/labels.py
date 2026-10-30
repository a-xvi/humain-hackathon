"""The 8 tools an Arabic AI assistant can route a request to. Single source of truth:
prompts, training and evaluation all read from here."""

LABELS = {
    "web_search": {
        "ar": "بحث في الإنترنت",
        "definition": "Needs current or real-time information from the internet: news, today's prices, weather, sports results, opening hours, 'who is the current ...'",
    },
    "image_generation": {
        "ar": "توليد صورة",
        "definition": "Create, draw or edit an image, logo, illustration, poster or design",
    },
    "code_assistant": {
        "ar": "مساعد برمجة",
        "definition": "Write, debug, explain or convert code, SQL queries, scripts, regex, Excel formulas",
    },
    "translation": {
        "ar": "ترجمة",
        "definition": "Translate text between languages (including translating a file or email the user gives)",
    },
    "math_calculation": {
        "ar": "حسابات",
        "definition": "Arithmetic, percentages, unit or currency-style conversions with given rates, equations, loan or budget calculations",
    },
    "document_qa": {
        "ar": "أسئلة عن ملف",
        "definition": "Summarize, extract from or answer questions about a document, PDF, contract, email or text the user provides (NOT translation)",
    },
    "scheduling": {
        "ar": "مواعيد وتذكيرات",
        "definition": "Set reminders, alarms, calendar events or meetings; check or change the user's schedule",
    },
    "general_chat": {
        "ar": "محادثة عامة",
        "definition": "Answer directly with no tool: general knowledge that doesn't change, advice, explanations, writing help, greetings, chit-chat",
    },
}

LABEL_LIST = list(LABELS)
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)
