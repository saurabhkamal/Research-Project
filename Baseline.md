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

⏱ Total Time: 13.25 seconds <br>
It took all together about 13.25 seconds to run inference on 20 examples.

📈 Throughput: 1.51 examples/sec<br>
That means BLIP-2 answered about: <br>
≈ 1.5 examples per second

🔍 Why This Script Runs Faster Than DePlot Versions
baseline notebook does only one inference per example:  BLIP-2(image + prompt) → answer

📌 Key Takeaways

✅ Baseline notebook is simple and fast — no table extraction.
✅ It’s using BLIP-2’s standard image+prompt generation.






