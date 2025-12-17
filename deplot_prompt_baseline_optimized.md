#### This optimized notebook is a refactored and faster version of your baseline pipeline that:

✔ Batches DePlot extraction and caches extracted tables to avoid re-doing expensive chart→table conversion for the same image.<br>
✔ Keeps the BLIP-2 visual-language answer generation step otherwise similar to baseline. <br>
✔ Produces performance metrics on a small test dataset with MC (multiple choice) and TF (true/false) questions.

The core idea remains the same:
1. Extract table text from chart images using DePlot
2. Build text + table prompts
3. Use BLIP-2 to answer questions
4. Evaluate accuracy

➤ DePlot (Chart → Table Extraction)

DePlot is a model that converts a chart image into a text representation of the underlying data table. It is part of the Pix2Struct family of vision-to-text structured models. DePlot focuses on chart derendering: taking plots (bar charts, line graphs, etc.) and generating a table text that captures the data points.

💡 What Changed in the Optimized Script
✅ 1. DePlot Caching
Before, every question would regenerate the table from the chart image, which is slow. Here:
A cache file (deplot_cache.pkl) stores earlier extracted table texts.

If an image was previously processed, the code reads the extracted table from the cache rather than recomputing it.

This dramatically cuts down repetitive calls to the expensive DePlot model.


Instead of processing one image at a time, the new script lets DePlot process multiple charts at once in a batch, which is much faster and more efficient on GPU hardware.

<img width="2158" height="210" alt="image" src="https://github.com/user-attachments/assets/fd1a6ce1-2166-4b50-ad70-3bd27e4b4d35" />

✔ Speed Metrics
- 47.56 seconds total for 20 examples → ~2.38 seconds per question.
- 0.42 examples/sec means less than half a question per second.

