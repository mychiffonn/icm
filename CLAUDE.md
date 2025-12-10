Re-implement the internal coherence maximization (ICM) algorithm from the paper "https://arxiv.org/html/2506.10139", without logical consistency fix. Read section 2.3 and algorithm 1 for details of mutual predictability algorithm.

Then, run the algorithm on TruthfulQA dataset in data/ folder.

## Set up

- `data` folder has truthfulqa_train.json and truthfulqa_test.json, of type list[dict[str, Any]]. Each dict has keys: 'question', 'choices', 'label', 'consistency_id'. `label` can be 0 and 1, where 1 indicates the truthful answer and 0 indicates the false answer.
- `results` folder to save evaluation results and figures.
- `src` folder has code files.
  - `src/__init__.py` can be empty, but I currently have base, chat model names, and generation config there. This information can be moved to `src/config.py` if needed.
  - `src/icm_search.py` should have the core implementation of ICM algorithm.
  - `src/utils.py` should have utility functions for loading data, prompting model, etc, to support `icm_search.py`
  - `src/eval.py` should have evaluation code. See details below.
  - `src/figure.py` should have code to generate figures. See details below.

## Evaluation

Four scenarios:

1. Zero-shot: Directly prompt the base model without any demonstrations for test set questions
2. Zero-shot Chat: Directly prompt the chat model without any demonstrations for test set questions
3. Golden Supervision: Few-shot prompting the base model with golden labels in `train` set, querying `test` set questions
4. ICM (ICL): Few-shot prompting the base model with ICM-predicted labels (predictions using `train` set), for test set questions

For each scenario, calculate the accuracy score on TruthfulQA test set and save results to `out` folder. Use exact match as the metric.

## Figure

Calculate the accuracy scores for all four scenarios on TruthfulQA test set, and plot a bar chart comparing the results. See TruthfulQA subchart of the following figure (Figure 1) from the paper.
![](https://arxiv.org/html/2506.10139/extracted/6534182/figures/fig1_llama.png)

The main figure should be output to `figures/truthfulqa_icm.png`.
