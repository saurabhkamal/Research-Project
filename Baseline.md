### This notebook runs a standard BLIP-2 pipeline

For each example:
1. It reads a chart image.
2. It builds a text prompt including the question.
3. It feeds the prompt + image to a BLIP-2 model.
4. It parses the model’s answer and compares it to the ground truth.
5. It records timing and accuracy.

Total examples: 20
🚀 Running inference …
Evaluating: 100%|██████████| 20/20 [00:13<00:00,  1.51example/s]
✅ Inference completed!
📊 MC Accuracy: 0.4
📊 TF Accuracy: 0.6
📊 Weighted Avg: 0.5
⏱ Total Time (s): 13.25
📈 Throughput (examples/sec): 1.51

