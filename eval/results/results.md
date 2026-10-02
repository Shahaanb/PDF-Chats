# Evaluation results

Run 2026-10-02 18:41 UTC · google `gemini-3.5-flash-lite` · embeddings `local BAAI/bge-small-en-v1.5` · chunks 1000/200

Retrieval: mmr, k = 6 · answer step sees the standalone question · grounding judge `gemini-3.7-flash` (falls back to the chat model)

Documents: attention_is_all_you_need.pdf, bert.pdf, rag_lewis_2020.pdf (50 pages, 227 chunks, indexed in 71.1 s)

**Summary:** correct 10/11 | retrieval hit@k 9/10 | cited 9/10 | grounded 11/11 | median latency 1.4s

| ID | Category | Question | Retrieval hit (rank) | Correct | Cited | Grounded (judge) | Latency |
|---|---|---|---|---|---|---|---|
| Q1 | Factual lookup | How many layers are in the Transformer's encoder stack? | yes (1) | yes | yes | grounded | 1.39 s |
| Q2 | Numeric fact | What BLEU score does the big Transformer achieve on the WMT 2014 English-to-German task? | yes (1) | yes | yes | grounded | 0.9 s |
| Q3 | Concept explanation | What is multi-head attention? | yes (1) | yes | yes | grounded | 1.83 s |
| Q4 | Follow-up (memory) | How many of them does the base model use, and what is the dimension of each? | yes (1) | yes | yes | grounded | 2.37 s |
| Q5 | Factual lookup | What are the two pre-training tasks used by BERT? | yes (1) | yes | yes | grounded | 1.08 s |
| Q6 | Follow-up (memory) | What percentage of tokens does it mask? | yes (1) | yes | yes | grounded | 1.9 s |
| Q7 | Comparison | What is the difference between RAG-Sequence and RAG-Token? | yes (4) | yes | yes | grounded | 1.21 s |
| Q8 | Factual lookup | What retriever and what generator does the RAG model use? | yes (1) | yes | yes | grounded | 1.38 s |
| Q9 | Cross-document | Which of these papers uses BERT inside its own architecture, and for what? | **no** | **no** | **no** | grounded | 1.05 s |
| Q10 | Table value | What was the training cost of the big Transformer model in FLOPs? | yes (1) | yes | yes | grounded | 119.59 s |
| Q11 | Out of scope | What dataset was GPT-3 trained on? | n/a | yes | n/a | grounded | 0.89 s |

## Answers

### Q1. How many layers are in the Transformer's encoder stack?

*Expected:* N = 6 identical layers

*Answer:*

> The encoder is composed of a stack of $N = 6$ identical layers [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.3, [2] attention_is_all_you_need.pdf p.9, [3] attention_is_all_you_need.pdf p.6, [4] attention_is_all_you_need.pdf p.2, [5] bert.pdf p.8, [6] attention_is_all_you_need.pdf p.2

*Judge:* grounded – The source passages explicitly state that the encoder is composed of a stack of N = 6 identical layers.

### Q2. What BLEU score does the big Transformer achieve on the WMT 2014 English-to-German task?

*Expected:* 28.4 BLEU

*Answer:*

> The big Transformer achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task [1][6].

*Retrieved:* [1] attention_is_all_you_need.pdf p.8, [2] rag_lewis_2020.pdf p.5, [3] bert.pdf p.6, [4] attention_is_all_you_need.pdf p.9, [5] attention_is_all_you_need.pdf p.8, [6] attention_is_all_you_need.pdf p.1

*Judge:* grounded – The answer is directly supported by the source text, which states that the big Transformer model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.

### Q3. What is multi-head attention?

*Expected:* Several attention functions (heads) run in parallel on different learned projections of queries, keys and values; their outputs are concatenated and projected

*Answer:*

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which counteracts the reduced effective resolution caused by averaging in single-head attention [1, 5]. 
> 
> Mathematically, multi-head attention consists of several attention layers running in parallel [1, 6]:
> $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
> where each head is calculated as:
> $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
> 
> The projections use the parameter matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ [1]. 
> 
> In the Transformer model, $h = 8$ parallel attention layers (or heads) are employed, using $d_k = d_v = d_{\text{model}}/h = 64$ for each head, which keeps the total computational cost similar to that of single-head attention with full dimensionality [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.5, [2] attention_is_all_you_need.pdf p.14, [3] attention_is_all_you_need.pdf p.3, [4] rag_lewis_2020.pdf p.15, [5] attention_is_all_you_need.pdf p.2, [6] attention_is_all_you_need.pdf p.4

*Judge:* grounded – All statements, formulas, and parameter definitions in the answer are directly derived from and supported by the provided source passages.

### Q4. How many of them does the base model use, and what is the dimension of each?

*Rewritten search query:* How many attention heads does the Transformer base model use, and what is the dimension of each head?

*Expected:* h = 8 heads, each with d_k = d_v = 64

*Answer:*

> The Transformer base model uses $h = 8$ parallel attention heads [1, 2]. The dimension of each head is $d_k = d_v = 64$ (which corresponds to $d_{\text{model}} / h$, where $d_{\text{model}} = 512$) [1, 2].

*Retrieved:* [1] attention_is_all_you_need.pdf p.5, [2] attention_is_all_you_need.pdf p.9, [3] bert.pdf p.8, [4] attention_is_all_you_need.pdf p.4, [5] attention_is_all_you_need.pdf p.5, [6] attention_is_all_you_need.pdf p.1

*Judge:* grounded – The source passages explicitly state that the base model uses 8 attention heads ($h=8$) and that the dimension of each head is $d_k = d_v = 64$ (with $d_{	ext{model}} = 512$).

### Q5. What are the two pre-training tasks used by BERT?

*Expected:* Masked language model (MLM) and next sentence prediction (NSP)

*Answer:*

> Based on the provided documents, the specific names or descriptions of the two pre-training tasks are not fully detailed in the text, though the passages mention that the model is trained on unlabeled data over different pre-training tasks [2] and note that "the two pre-training tasks presented in Section 3.1 account for the majority of the empirical improvements" [3]. (Figure 1 references NSP and Mask LM during pre-training [1]).

*Retrieved:* [1] bert.pdf p.3, [2] bert.pdf p.3, [3] bert.pdf p.14, [4] bert.pdf p.1, [5] bert.pdf p.5, [6] bert.pdf p.1

*Judge:* grounded – The provided passages mention that BERT uses two pre-training tasks (referenced as NSP and Mask LM in Figure 1, though Section 3.1 itself is not included in full), and the answer accurately reflects what is and isn't detailed in the text.

### Q6. What percentage of tokens does it mask?

*Rewritten search query:* What percentage of tokens does BERT mask during its pre-training?

*Expected:* 15% of WordPiece tokens (then 80% [MASK], 10% random, 10% unchanged)

*Answer:*

> During BERT pre-training, the training data generator chooses **15%** of the token positions at random for prediction [5].

*Retrieved:* [1] bert.pdf p.12, [2] bert.pdf p.6, [3] bert.pdf p.16, [4] bert.pdf p.5, [5] bert.pdf p.4, [6] bert.pdf p.4

*Judge:* grounded – The answer is directly supported by passage [5], which states that the training data generator chooses 15% of the token positions at random for prediction.

### Q7. What is the difference between RAG-Sequence and RAG-Token?

*Expected:* RAG-Sequence uses the same retrieved document for the whole output; RAG-Token can use a different document for each token

*Answer:*

> Based on the provided documents, the differences between the RAG-Sequence and RAG-Token models are:
> 
> * **RAG-Sequence Model:** This model treats the retrieved document as a single latent variable that is marginalized to get the sequence-to-sequence probability $p(y|x)$ via a top-$K$ approximation [4]. This means the same retrieved document is used across the entire generated sequence [4]. Additionally, retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence [5].
> * **RAG-Token Model:** In this model, a different latent document can be drawn for each target token and marginalized accordingly [4]. This allows the generator to choose content from several different documents when producing an answer [4]. For Open-domain QA, performance for RAG-Token peaks at 10 retrieved documents rather than monotonically improving [5].

*Retrieved:* [1] rag_lewis_2020.pdf p.6, [2] rag_lewis_2020.pdf p.17, [3] bert.pdf p.4, [4] rag_lewis_2020.pdf p.3, [5] rag_lewis_2020.pdf p.8, [6] rag_lewis_2020.pdf p.6

*Judge:* grounded – All claims in the answer regarding the differences between RAG-Sequence and RAG-Token are directly supported by source passages [4] and [5].

### Q8. What retriever and what generator does the RAG model use?

*Expected:* DPR (bi-encoder with BERT-base encoders) as retriever, BART-large as generator

*Answer:*

> Based on the provided documents, the RAG model uses the following components:
> 
> * **Retriever:** RAG leverages a retriever $p_\eta(z|x)$ with parameters $\eta$ that returns (top-K truncated) distributions over text passages given a query $x$ [1]. The retriever is initialized using DPR's retriever, which uses retrieval supervision on Natural Questions and TriviaQA [3]. 
> * **Generator:** RAG uses a generator $p_\theta(y_i | x, z, y_{1:i-1})$ parametrized by BART generator parameters $\theta$ (referred to as the parametric memory), which are pre-trained using a denoising objective and a variety of different noising functions [1, 2].

*Retrieved:* [1] rag_lewis_2020.pdf p.2, [2] rag_lewis_2020.pdf p.3, [3] rag_lewis_2020.pdf p.5, [4] rag_lewis_2020.pdf p.8, [5] rag_lewis_2020.pdf p.7, [6] rag_lewis_2020.pdf p.9

*Judge:* grounded – All claims regarding the retriever (including its mathematical notation and DPR initialization) and the generator (including its mathematical formulation and use of BART pre-trained parameters) are directly supported by the provided passages.

### Q9. Which of these papers uses BERT inside its own architecture, and for what?

*Expected:* The RAG paper: its DPR retriever uses BERT-base query and document encoders

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.14, [2] bert.pdf p.7, [3] bert.pdf p.10, [4] bert.pdf p.4, [5] bert.pdf p.3, [6] bert.pdf p.12

*Judge:* grounded – The source passages do not contain information about other papers using BERT inside their architecture, so the answer correctly states that the information cannot be found in the provided documents.

### Q10. What was the training cost of the big Transformer model in FLOPs?

*Expected:* 2.3 x 10^19 FLOPs (Table 2)

*Answer:*

> Based on Table 2, the training cost for the Transformer (big) model is $2.3 \cdot 10^{19}$ FLOPs for English-to-German (EN-DE) [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.8, [2] attention_is_all_you_need.pdf p.7, [3] bert.pdf p.8, [4] rag_lewis_2020.pdf p.3, [5] attention_is_all_you_need.pdf p.8, [6] attention_is_all_you_need.pdf p.3

*Judge:* grounded – Table 1 (cited as Table 2 in the source) explicitly states that the training cost for the Transformer (big) model is 2.3 · 10^19 FLOPs for English-to-German (EN-DE).

### Q11. What dataset was GPT-3 trained on?

*Expected:* Not in the documents: the assistant should say it could not find this

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.15, [2] attention_is_all_you_need.pdf p.7, [3] attention_is_all_you_need.pdf p.7, [4] bert.pdf p.3, [5] attention_is_all_you_need.pdf p.2, [6] rag_lewis_2020.pdf p.3

*Judge:* grounded – The source passages do not contain information about the dataset that GPT-3 was trained on, so the assistant's response stating that the information is not present is correct.
