"""The 8 departments. Single source of truth: prompts, training and evaluation all read from here."""

LABELS = {
    "orders_delivery": {
        "ar": "الطلبات والتوصيل",
        "definition": "Order status, delivery delays, wrong address, courier issues",
    },
    "returns_refunds": {
        "ar": "الإرجاع واسترداد المبالغ",
        "definition": "Returning or exchanging a product, refunds for returned products, return/refund status",
    },
    "payments_billing": {
        "ar": "المدفوعات والفواتير",
        "definition": "Payment failures, wrong or duplicate charges (including refunds for them), invoices, payment methods, installments",
    },
    "account_login": {
        "ar": "الحساب وتسجيل الدخول",
        "definition": "Login problems, password resets, OTP/verification codes, account access, registration, updating account details",
    },
    "app_technical": {
        "ar": "المشاكل التقنية والتطبيق",
        "definition": "App or website crashes, bugs, loading errors, technical malfunctions (NOT login or password problems)",
    },
    "product_inquiry": {
        "ar": "الاستفسارات عن المنتجات",
        "definition": "Questions before buying: availability, prices, specifications, sizes, product information",
    },
    "service_complaint": {
        "ar": "الشكاوى على جودة الخدمة",
        "definition": "Complaints about staff behaviour or overall service quality, NOT about a specific order's delivery",
    },
    "other": {
        "ar": "أخرى",
        "definition": "Messages that fit no other department: general questions, suggestions, greetings, unrelated messages",
    },
}

LABEL_LIST = list(LABELS)
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)
