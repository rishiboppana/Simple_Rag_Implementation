# Evaluation summary

embedding=BAAI/bge-base-en-v1.5; chunking=semantic + 800/120 tokens; LLM=Qwen/Qwen3-8B; judge=Qwen/Qwen3-32B; min_relevance=0.35

| Experiment | Recall@k | Correctness | Citation | Groundedness | Abstention | Unjustified refusals |
|---|---|---|---|---|---|---|
| baseline_k4 | 83% | 79% | 82% | 92% | 100% | 2 |
| k2 | 79% | 75% | 77% | 83% | 100% | 2 |
| k8 | 88% | 77% | 86% | 92% | 100% | 2 |
| rerank_k4 | 92% | 79% | 91% | 96% | 100% | 2 |
| hyde_k4 | 71% | 73% | 81% | 88% | 100% | 3 |
| rerank_k8 | 96% | 77% | 95% | 92% | 83% | 2 |
| rerank_k4_nogate | 92% | 83% | 92% | 88% | 100% | 0 |
| rerank_k8_nogate | 96% | 88% | 96% | 92% | 83% | 0 |
| parent_m1_k4_nogate | 71% | 69% | 77% | 88% | 100% | 2 |
| parent_m3_k4_nogate | 79% | 71% | 79% | 79% | 100% | 0 |
| parent_m3_k8_nogate | 79% | 85% | 79% | 96% | 83% | 0 |
| parent_m5_k8_nogate | 83% | 88% | 83% | 92% | 100% | 0 |
