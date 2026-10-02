# Evaluation results: first run

Our first run, with the baseline settings and the chat model as its own judge. Kept because it shows the table-value failure (Q10) that later runs did not repeat.

Run 2026-10-02 18:26 UTC · google `gemini-3.5-flash-lite` · embeddings `local BAAI/bge-small-en-v1.5` · chunks 1000/200

Retrieval: similarity, k = 4 · answer step sees the raw question · grounding judge `gemini-3.5-flash-lite (same as chat model)`

Documents: attention_is_all_you_need.pdf, bert.pdf, rag_lewis_2020.pdf (50 pages, 227 chunks, indexed in 69.5 s)

**Summary:** correct 6/11 | retrieval hit@k 9/10 | cited 7/10 | grounded 8/11 | median latency 3.2s

| ID | Category | Question | Retrieval hit (rank) | Correct | Cited | Grounded (judge) | Latency |
|---|---|---|---|---|---|---|---|
| Q1 | Factual lookup | How many layers are in the Transformer's encoder stack? | yes (1) | yes | yes | grounded | 1.59 s |
| Q2 | Numeric fact | What BLEU score does the big Transformer achieve on the WMT 2014 English-to-German task? | yes (1) | yes | yes | grounded | 1.43 s |
| Q3 | Concept explanation | What is multi-head attention? | yes (1) | yes | yes | grounded | 54.17 s |
| Q4 | Follow-up (memory) | How many of them does the base model use, and what is the dimension of each? | yes (1) | **no** | **no** | not grounded | 40.89 s |
| Q5 | Factual lookup | What are the two pre-training tasks used by BERT? | yes (1) | yes | yes | grounded | 3.18 s |
| Q6 | Follow-up (memory) | What percentage of tokens does it mask? | yes (1) | **no** | **no** | not grounded | 7.45 s |
| Q7 | Comparison | What is the difference between RAG-Sequence and RAG-Token? | yes (2) | yes | yes | grounded | 4.37 s |
| Q8 | Factual lookup | What retriever and what generator does the RAG model use? | yes (1) | **no** | yes | grounded | 2.31 s |
| Q9 | Cross-document | Which of these papers uses BERT inside its own architecture, and for what? | **no** | **no** | **no** | grounded | 3.24 s |
| Q10 | Table value | What was the training cost of the big Transformer model in FLOPs? | yes (1) | **no** | yes | not grounded | 1.63 s |
| Q11 | Out of scope | What dataset was GPT-3 trained on? | n/a | yes | n/a | grounded | 2.04 s |

## Answers

### Q1. How many layers are in the Transformer's encoder stack?

*Expected:* N = 6 identical layers

*Answer:*

> The encoder is composed of a stack of $N = 6$ identical layers [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.3, [2] attention_is_all_you_need.pdf p.5, [3] attention_is_all_you_need.pdf p.3, [4] attention_is_all_you_need.pdf p.2

*Judge:* grounded – The source passages explicitly state that the encoder is composed of a stack of N = 6 identical layers.

### Q2. What BLEU score does the big Transformer achieve on the WMT 2014 English-to-German task?

*Expected:* 28.4 BLEU

*Answer:*

> The big Transformer achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task [1][3][4].

*Retrieved:* [1] attention_is_all_you_need.pdf p.8, [2] attention_is_all_you_need.pdf p.8, [3] attention_is_all_you_need.pdf p.8, [4] attention_is_all_you_need.pdf p.1

*Judge:* grounded – The source passages explicitly state that the big Transformer model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.

### Q3. What is multi-head attention?

*Expected:* Several attention functions (heads) run in parallel on different learned projections of queries, keys and values; their outputs are concatenated and projected

*Answer:*

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which counteracts the reduced effective resolution caused by averaging attention-weighted positions [1][4]. 
> 
> Mathematically, Multi-Head Attention is defined as:
> $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,..., \text{head}_h)W^O$$
> where each head is computed as:
> $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
> 
> The projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{model}}$ [1]. In the described work, $h = 8$ parallel attention layers (or heads) are employed, using $d_k = d_v = d_{model}/h = 64$ [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.5, [2] attention_is_all_you_need.pdf p.5, [3] attention_is_all_you_need.pdf p.4, [4] attention_is_all_you_need.pdf p.2

*Judge:* grounded – All the statements, mathematical formulas, definitions, and specific parameter values provided in the answer are directly supported by the text in the source passages.

### Q4. How many of them does the base model use, and what is the dimension of each?

*Rewritten search query:* How many attention heads does the Transformer base model use, and what is the dimension of each head?

*Expected:* h = 8 heads, each with d_k = d_v = 64

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] attention_is_all_you_need.pdf p.5, [2] attention_is_all_you_need.pdf p.5, [3] attention_is_all_you_need.pdf p.3, [4] bert.pdf p.3

*Judge:* not grounded – The source passages do contain the answer: passage [2] states that they employ h = 8 parallel attention layers or heads, and $d_k = d_v = d_{model}/h = 64$ for the dimension of each head.

### Q5. What are the two pre-training tasks used by BERT?

*Expected:* Masked language model (MLM) and next sentence prediction (NSP)

*Answer:*

> Based on the provided documents, the two pre-training tasks used by BERT are Masked LM (MLM) and Next Sentence Prediction (NSP) [1][2].

*Retrieved:* [1] bert.pdf p.3, [2] bert.pdf p.14, [3] bert.pdf p.14, [4] bert.pdf p.2

*Judge:* grounded – The source passages mention 'NSP' and 'Mask LM' under pre-training in Figure 1, which correspond to the two pre-training tasks.

### Q6. What percentage of tokens does it mask?

*Rewritten search query:* What percentage of tokens does BERT mask during its pre-training?

*Expected:* 15% of WordPiece tokens (then 80% [MASK], 10% random, 10% unchanged)

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.12, [2] bert.pdf p.3, [3] bert.pdf p.16, [4] bert.pdf p.5

*Judge:* not grounded – The provided source passages do not mention the percentage of tokens BERT masks during pre-training.

### Q7. What is the difference between RAG-Sequence and RAG-Token?

*Expected:* RAG-Sequence uses the same retrieved document for the whole output; RAG-Token can use a different document for each token

*Answer:*

> The difference between the two models lies in how they handle retrieved documents to produce the distribution over generated text:
> 
> * **RAG-Sequence Model:** This model uses the same retrieved document to generate the complete target sequence [2]. It treats the retrieved document as a single latent variable that is marginalized to obtain the seq2seq probability $p(y|x)$ via a top-K approximation [2].
> * **RAG-Token Model:** This model can predict each target token based on a different retrieved document [2].

*Retrieved:* [1] rag_lewis_2020.pdf p.6, [2] rag_lewis_2020.pdf p.3, [3] rag_lewis_2020.pdf p.2, [4] rag_lewis_2020.pdf p.6

*Judge:* grounded – The assistant's explanation of the differences between RAG-Sequence and RAG-Token is directly supported by the text in source [2].

### Q8. What retriever and what generator does the RAG model use?

*Expected:* DPR (bi-encoder with BERT-base encoders) as retriever, BART-large as generator

*Answer:*

> Based on the provided document, the RAG model leverages two components:
> 1. A retriever $p_\eta(z|x)$ with parameters $\eta$ that returns (top-K truncated) distributions over text passages given a query $x$ [1].
> 2. A generator parametrized by $\theta$ (denoted as $p_\theta(y_i | x, z, y_{1:i-1})$) that generates a current token based on a context of the previous $i-1$ tokens $y_{1:i-1}$, the original input $x$, and a retrieved passage $z$ [1][2].

*Retrieved:* [1] rag_lewis_2020.pdf p.2, [2] rag_lewis_2020.pdf p.3, [3] rag_lewis_2020.pdf p.7, [4] rag_lewis_2020.pdf p.9

*Judge:* grounded – The answer accurately describes the retriever and generator components of the RAG model using the exact definitions and mathematical notations provided in the source passages.

### Q9. Which of these papers uses BERT inside its own architecture, and for what?

*Expected:* The RAG paper: its DPR retriever uses BERT-base query and document encoders

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.14, [2] bert.pdf p.9, [3] bert.pdf p.2, [4] bert.pdf p.1

*Judge:* grounded – The provided source passages discuss BERT and its pre-training/fine-tuning capabilities, but they do not mention any specific papers using BERT inside their own architecture. Therefore, stating that the information could not be found is correct.

### Q10. What was the training cost of the big Transformer model in FLOPs?

*Expected:* 2.3 x 10^19 FLOPs (Table 2)

*Answer:*

> Based on the provided document, the training cost (in FLOPs) for the Transformer (big) model is not explicitly listed in Table 2 for English-to-French, but Table 2 lists the English-to-German and English-to-French training costs for the "Transformer (base model)" as $3.3 \cdot 10^{18}$ [1]. 
> 
> For the English-to-French task specifically, the document states that the big model achieves its results "at less than 1/4 the training cost of the previous state-of-the-art model" [3], but it does not state the exact FLOPs value for the Transformer (big) model in the text or tables. 
> 
> I couldn't find this specific FLOPs value for the big Transformer model in the uploaded documents.

*Retrieved:* [1] attention_is_all_you_need.pdf p.8, [2] attention_is_all_you_need.pdf p.8, [3] attention_is_all_you_need.pdf p.8, [4] attention_is_all_you_need.pdf p.8

*Judge:* not grounded – Table 2 explicitly lists the training cost in FLOPs for the Transformer (big) model as 2.3 × 10^19 (shown in the row for Transformer (big) under the EN-DE column, which applies to the model's training cost as summarized in the table). Therefore, the assistant's claim that the value is not in the document is incorrect.

### Q11. What dataset was GPT-3 trained on?

*Expected:* Not in the documents: the assistant should say it could not find this

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.15, [2] bert.pdf p.6, [3] bert.pdf p.13, [4] bert.pdf p.14

*Judge:* grounded – The provided source passages mention OpenAI GPT and discuss its training on a large text corpus, but they do not contain information specifically about GPT-3 and the dataset it was trained on.
