#### This notebook implements an automated pipeline to answer questions about chart images, using:
1. DePlot — to convert chart images into data tables; and
2. BLIP-2 — a multimodal vision-language model to answer questions about those tables.

Then it evaluates performance on multiple-choice (MC) and true/false (TF) questions.

📊 DePlot Extraction Helper <br>
DePlot is a specialized chart table extractor — it converts chart images into a text description of the underlying data table. It uses the Pix2Struct architecture.

Total examples: 20 <br>
🚀 Running inference … <br>
Evaluating: 100%|██████████| 20/20 [02:07<00:00,  6.36s/example] <br>
✅ Inference completed! <br>
📊 MC Accuracy: 0.2 <br>
📊 TF Accuracy: 0.6 <br>
📊 Weighted Avg: 0.4 <br>
⏱ Total Time (s): 127.23 <br>
📈 Throughput (examples/sec): 0.16

~127s total: Slow extraction + model inference <br>
0.16 examples/sec: ~1 answer every 6 seconds
