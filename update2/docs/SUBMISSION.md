# Submission text

## Project title
Sikkah (سكة): Offline Arabic Tool Router for AI Assistants

## Challenge category
Challenge 2: Deliver AI with Less Compute

## Short description (one line)
A small Arabic model that decides which tool an AI assistant should use, offline on a laptop CPU, in place of a full LLM call. It is trained with active distillation, so it also needs fewer GPT labels.

## Project description
Before an AI assistant does anything, it decides which tool should handle the request: web search, image generation, code, translation, math, a document, the calendar, or plain chat. Today that decision is usually a full LLM call, which means a GPU, internet access and a paid API call on every request.

I distilled that decision from GPT-4.1-mini into MARBERTv2, a small Arabic encoder that runs offline on one CPU thread. It handles Najdi, Hijazi, Eastern and MSA Arabic, plus Arabizi and mixed English. Instead of paying GPT to label every training example, the router uses active distillation: it picks the requests it is least sure about, and only those get labeled.

Results on 320 held-out requests whose labels were checked by a person:
- **Macro-F1:** 0.944, vs 0.918 for GPT-4.1-mini. The confidence intervals overlap, so the two match.
- **Latency:** 101 ms vs 707 ms (7× faster), with no internet and no cost per request.
- **Training labels:** 640 GPT labels instead of 1,920. Margin sampling at 480 labels beat random sampling at 640.
- **Hybrid mode:** sending only the 9.7% least-confident requests to GPT raises macro-F1 to 0.975, higher than either model alone.

I also tested robustness:
- Unseen dialects (Egyptian, Levantine): 0.949.
- Arabizi went from 0.156 to 0.621 through augmentation, with no extra API calls. It still trails GPT, and the augmentation reduced mixed-English accuracy. Both are reported openly.
- int8 quantization was 3× faster but lost 0.10 macro-F1, so I rejected it and ship fp32.

## Links
- GitHub: https://github.com/a-xvi/humain-hackathon
- App/code: https://github.com/a-xvi/humain-hackathon (run `python app.py`; works offline)
- Demo video: <add link>
- Presentation: <add link or PDF>
