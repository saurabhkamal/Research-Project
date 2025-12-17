### This notebook runs a standard BLIP-2 pipeline

For each example:
1. It reads a chart image.
2. It builds a text prompt including the question.
3. It feeds the prompt + image to a BLIP-2 model.
4. It parses the model’s answer and compares it to the ground truth.
5. It records timing and accuracy.

Total examples: 20 <br>
🚀 Running inference … <br>
Evaluating: 100%|██████████| 20/20 [00:13<00:00,  1.51example/s] <br>
✅ Inference completed! <br>
📊 MC Accuracy: 0.4 <br>
📊 TF Accuracy: 0.6 <br>
📊 Weighted Avg: 0.5 <br>
⏱ Total Time (s): 13.25 <br>
📈 Throughput (examples/sec): 1.51 <br>

