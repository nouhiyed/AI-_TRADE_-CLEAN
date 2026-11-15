# AI-Trader — Scalping V3

نسخة التطبيق المعتمدة للسكالبينغ فقط.

## المكونات
- Futures Agent: استراتيجية Scalping V3، اتجاه LONG وSHORT.
- Spot Agent: استراتيجية Scalping V3، اتجاه LONG فقط.
- بيانات السوق من Binance عبر الإنترنت.
- وضع DEMO بمحاكاة داخلية ورصيد افتراضي.
- LIVE ظاهر في التطبيق لكنه مقفل في الخلفية.
- Autopilot متوقف افتراضياً؛ لا يبدأ التداول التلقائي دون تفعيله من داخل التطبيق.

## تشغيل Backend على Termux جديد
```bash
pkg update -y && pkg upgrade -y
pkg install -y git python python-numpy python-pandas
cd ~
git clone https://github.com/nouhiyed/AI-_TRADE_-CLEAN.git AI-Trader
cd ~/AI-Trader
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m api.main
```

الخادم المحلي يعمل على المنفذ `8001`. يجب إبقاء الإنترنت متاحاً لجلب بيانات Binance.

## بناء التطبيق
من GitHub افتح Actions ثم **Build AI Trade APK** وشغّل workflow على الفرع `main`. سينتج ملف APK باسم `AI-Trade-APK`.

## تنبيه
هذه النسخة لا تحتوي ملفات نتائج أو أدوات backtest أو مجلدات اختبارات أو بيانات تشغيل سابقة. لا تنسخ ملفات الإعدادات وقواعد البيانات من تثبيت Termux القديم إلى التثبيت الجديد. ابدأ DEMO برصيد افتراضي، واترك LIVE مقفلاً.
