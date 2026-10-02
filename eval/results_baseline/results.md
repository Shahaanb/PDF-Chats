# Evaluation results

Run 2026-10-02 18:36 UTC · google `gemini-3.5-flash-lite` · embeddings `local BAAI/bge-small-en-v1.5` · chunks 1000/200

Retrieval: similarity, k = 4 · answer step sees the raw question · grounding judge `gemini-3.7-flash` (falls back to the chat model)

Documents: attention_is_all_you_need.pdf, bert.pdf, rag_lewis_2020.pdf (50 pages, 227 chunks, indexed in 80.2 s)

**Summary:** correct 7/11 | retrieval hit@k 9/10 | cited 7/10 | grounded 9/11 | median latency 1.2s

| ID | Category | Question | Retrieval hit (rank) | Correct | Cited | Grounded (judge) | Latency |
|---|---|---|---|---|---|---|---|
| Q1 | Factual lookup | How many layers are in the Transformer's encoder stack? | yes (1) | yes | yes | grounded | 1.7 s |
| Q2 | Numeric fact | What BLEU score does the big Transformer achieve on the WMT 2014 English-to-German task? | yes (1) | yes | yes | grounded | 1.24 s |
| Q3 | Concept explanation | What is multi-head attention? | yes (1) | yes | yes | grounded | 1.68 s |
| Q4 | Follow-up (memory) | How many of them does the base model use, and what is the dimension of each? | yes (1) | **no** | **no** | not grounded | 1.57 s |
| Q5 | Factual lookup | What are the two pre-training tasks used by BERT? | yes (1) | yes | yes | grounded | 1.06 s |
| Q6 | Follow-up (memory) | What percentage of tokens does it mask? | yes (1) | **no** | **no** | not grounded | 1.8 s |
| Q7 | Comparison | What is the difference between RAG-Sequence and RAG-Token? | yes (2) | yes | yes | grounded | 1.63 s |
| Q8 | Factual lookup | What retriever and what generator does the RAG model use? | yes (1) | **no** | yes | grounded | 1.17 s |
| Q9 | Cross-document | Which of these papers uses BERT inside its own architecture, and for what? | **no** | **no** | **no** | grounded | 0.87 s |
| Q10 | Table value | What was the training cost of the big Transformer model in FLOPs? | yes (1) | yes | yes | grounded | 1.24 s |
| Q11 | Out of scope | What dataset was GPT-3 trained on? | n/a | yes | n/a | grounded | 1.21 s |

## Answers

### Q1. How many layers are in the Transformer's encoder stack?

*Expected:* N = 6 identical layers

*Answer:*

> The encoder stack in the Transformer is composed of $N = 6$ identical layers [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.3, [2] attention_is_all_you_need.pdf p.5, [3] attention_is_all_you_need.pdf p.3, [4] attention_is_all_you_need.pdf p.2

*Judge:* grounded – The source passages explicitly state that the encoder is composed of a stack of N = 6 identical layers.

### Q2. What BLEU score does the big Transformer achieve on the WMT 2014 English-to-German task?

*Expected:* 28.4 BLEU

*Answer:*

> The big Transformer model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task [1][3].

*Retrieved:* [1] attention_is_all_you_need.pdf p.8, [2] attention_is_all_you_need.pdf p.8, [3] attention_is_all_you_need.pdf p.8, [4] attention_is_all_you_need.pdf p.1

*Judge:* grounded – The provided passages explicitly state that the big Transformer model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.

### Q3. What is multi-head attention?

*Expected:* Several attention functions (heads) run in parallel on different learned projections of queries, keys and values; their outputs are concatenated and projected

*Answer:*

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which counteracts the reduced effective resolution caused by averaging attention-weighted positions [1][4]. 
> 
> Mathematically, Multi-Head Attention is defined as:
> $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$
> where each head is computed as:
> $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
> 
> The projections are parameter matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ [1]. 
> 
> In the Transformer, $h = 8$ parallel attention layers (or heads) are employed, using $d_k = d_v = d_{\text{model}}/h = 64$ for each [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.5, [2] attention_is_all_you_need.pdf p.5, [3] attention_is_all_you_need.pdf p.4, [4] attention_is_all_you_need.pdf p.2

*Judge:* grounded – All parts of the answer are directly supported by the provided source passages.

### Q4. How many of them does the base model use, and what is the dimension of each?

*Rewritten search query:* How many multi-head attention layers does the Transformer base model use, and what is the dimension of each head?

*Expected:* h = 8 heads, each with d_k = d_v = 64

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] attention_is_all_you_need.pdf p.5, [2] attention_is_all_you_need.pdf p.5, [3] attention_is_all_you_need.pdf p.3, [4] bert.pdf p.3

*Judge:* not grounded – The source passages do contain the information: passage [1] states that they employ h = 8 parallel attention layers or heads, and that for each of these dk = dv = 64.

### Q5. What are the two pre-training tasks used by BERT?

*Expected:* Masked language model (MLM) and next sentence prediction (NSP)

*Answer:*

> Based on the provided documents, the two pre-training tasks used by BERT are Mask LM (MLM) and NSP (Next Sentence Prediction) [1][2].

*Retrieved:* [1] bert.pdf p.3, [2] bert.pdf p.14, [3] bert.pdf p.14, [4] bert.pdf p.2

*Judge:* grounded – Passage 1 explicitly labels the pre-training tasks as 'NSP' and 'Mask LM' (and the answer correctly identifies them based on the text).

### Q6. What percentage of tokens does it mask?

*Rewritten search query:* What percentage of tokens does BERT mask during its pre-training?

*Expected:* 15% of WordPiece tokens (then 80% [MASK], 10% random, 10% unchanged)

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.12, [2] bert.pdf p.3, [3] bert.pdf p.16, [4] bert.pdf p.5

*Judge:* not grounded – The provided source passages do not state the percentage of tokens BERT masks during its pre-training, making the assistant's claim that the documents do not contain the information correct. However, the evaluation framework requires the verdict to be "not grounded" when the information is indeed missing from the passages (as per standard instruction: "An answer that says the documents do not contain the information is grounded when the passages really do not contain it", but the schema specifically defines 'grounded' as when every claim is supported by the source passages. Wait, let's re-read: "An answer that says the documents do not contain the information is grounded when the passages really do not contain it." Thus, since the passages do not contain the info, and the answer says it could not find it, the verdict should be grounded.) Let me correct the verdict to grounded.

### Q7. What is the difference between RAG-Sequence and RAG-Token?

*Expected:* RAG-Sequence uses the same retrieved document for the whole output; RAG-Token can use a different document for each token

*Answer:*

> Based on the provided documents, the difference between the two models lies in how they handle retrieved documents to produce the distribution over generated text:
> 
> * **RAG-Sequence Model:** Uses the same retrieved document to generate the complete sequence, treating the retrieved document as a single latent variable that is marginalized to get the seq2seq probability via a top-K approximation [2].
> * **RAG-Token Model:** Can predict each target token based on a different retrieved document [2].

*Retrieved:* [1] rag_lewis_2020.pdf p.6, [2] rag_lewis_2020.pdf p.3, [3] rag_lewis_2020.pdf p.2, [4] rag_lewis_2020.pdf p.6

*Judge:* grounded – The answer accurately reflects the distinction between RAG-Sequence and RAG-Token as directly stated in passage [2].

### Q8. What retriever and what generator does the RAG model use?

*Expected:* DPR (bi-encoder with BERT-base encoders) as retriever, BART-large as generator

*Answer:*

> The RAG model leverages two components:
> 1. A retriever $p_\eta(z|x)$ with parameters $\eta$ that returns (top-K truncated) distributions over text passages given a query $x$ [1].
> 2. A generator parametrized by $\theta$ (denoted as $p_\theta(y_i | x, z, y_{1:i-1})$) that generates a current token based on the original input $x$, a retrieved passage $z$, and the context of the previous $i-1$ tokens $y_{1:i-1}$ [1][2].

*Retrieved:* [1] rag_lewis_2020.pdf p.2, [2] rag_lewis_2020.pdf p.3, [3] rag_lewis_2020.pdf p.7, [4] rag_lewis_2020.pdf p.9

*Judge:* grounded – The answer accurately describes the retriever and generator components of the RAG model as stated in the source passages.

### Q9. Which of these papers uses BERT inside its own architecture, and for what?

*Expected:* The RAG paper: its DPR retriever uses BERT-base query and document encoders

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.14, [2] bert.pdf p.9, [3] bert.pdf p.2, [4] bert.pdf p.1

*Judge:* grounded – The provided source passages do not contain information about any papers that use BERT inside their own architecture, so the assistant correctly stated that it could not find this information.

### Q10. What was the training cost of the big Transformer model in FLOPs?

*Expected:* 2.3 x 10^19 FLOPs (Table 2)

*Answer:*

> Based on Table 2, the training cost of the Transformer (big) model is $2.3 \cdot 10^{19}$ FLOPs [1].

*Retrieved:* [1] attention_is_all_you_need.pdf p.8, [2] attention_is_all_you_need.pdf p.8, [3] attention_is_all_you_need.pdf p.8, [4] attention_is_all_you_need.pdf p.8

*Judge:* grounded – The answer accurately reflects Table 2 in the source passages, which lists the training cost of the Transformer (big) model as 2.3 · 10^19 FLOPs.

### Q11. What dataset was GPT-3 trained on?

*Expected:* Not in the documents: the assistant should say it could not find this

*Answer:*

> I couldn't find this in the uploaded documents.

*Retrieved:* [1] bert.pdf p.15, [2] bert.pdf p.6, [3] bert.pdf p.13, [4] bert.pdf p.14

*Judge:* grounded – The provided source passages discuss OpenAI GPT, BERT, and various datasets like WNLI, but they contain no information about what dataset GPT-3 was trained on.
