#### This script performs chart question answering using a DePlot → BLIP-2 pipeline:
1. DePlot (Pix2Struct) extracts a text table from each chart image.
   - DePlot does not cache results in this version — it runs DePlot every time an image is processed.
2. The extracted table is combined with the question text to form a prompt.
3. BLIP-2 (Salesforce/blip2-opt-2.7b) then answers the question based on the image + prompt.
4. The answers are normalized and evaluated against ground truth to compute accuracy.

This is the “optimized without cache” version — so it batch-extracts tables but doesn’t use a cache.

Total examples: 20
🚀 Running inference …
...
📊 MC Accuracy: 0.2
📊 TF Accuracy: 0.4
📊 Weighted Avg: 0.3
⏱ Total Time (s): 78.05
📈 Throughput (examples/sec): 0.26

⏱ Performance / Speed
- Total Time (s): 78.05: ~78 seconds to process all 20 images
- Avg Time per Question: ~3.90 s: Roughly ~4 seconds per inference
- Throughput: 0.26 examples/sec: ~1 question every 4 seconds

This is faster than the baseline (~6+ seconds/example) because:
✔ Batch DePlot (where possible)
✔ BLIP-2 in optimized configuration with GPU

But slower than the cached variant (which stores previous table extractions).


| Feature          | Baseline      | Optimized with Cache | Optimized without Cache |
| ---------------- | ------------- | -------------------- | ----------------------- |
| Batch DePlot     | ❌             | ✔                    | ✔                       |
| Caching          | ❌             | ✔                    | ❌                       |
| BLIP-2 Prompting | ✔             | ✔                    | ✔                       |
| Speed            | Slow          | Fastest              | Faster than baseline    |
| Accuracy         | ~40% weighted | ~40% weighted        | ~30% weighted           |



