# Overlap Analysis: DH2022 vs CL Journal Submission

## Executive Summary

**Can this be published?** YES - with proper acknowledgment and emphasis on what's new.

The DH2022 presentation was a **proof-of-concept** showing USE could detect adaptations.
The CL journal paper is a **systematic benchmark evaluation** of 15 models across 3 architectures.

**Recommendation:** Acknowledge DH2022 in paper, retitle to emphasize benchmark contribution, and highlight the 80% new content.

---

## What Was in DH2022 (2022)

### Core Elements
- **Dataset**: HathiTrust Crusoe adaptations (1,484) + ECCO-TCP random texts (2,188)
- **Method**: USE embeddings + binary classifier
- **Research Question**: "Can ML find Robinson Crusoe adaptations?"
- **Models Tested**: 3 models
  - TF-IDF + cosine similarity
  - BERT embeddings
  - Universal Sentence Encoder (USE)
- **Result**: Near-perfect accuracy with USE
- **Contribution**: Novel pipeline for adaptation detection

### Key Claims (DH2022)
1. USE embeddings work better than TF-IDF and BERT for plot-level similarity
2. The model can implicitly learn plot structure
3. High accuracy achieved (~99%)

### Framing (DH2022)
- **Focus**: Digital Humanities application
- **Goal**: Find undiscovered Crusoe adaptations
- **Audience**: Literary scholars and DH community
- **Contribution Type**: Methodological tool for literary analysis

---

## What's NEW in CL Journal Paper (2025)

### Massive Expansion (80% new content)

#### 1. **Systematic Model Comparison** (ENTIRELY NEW)
- **15 models evaluated** (vs 3 in DH2022)
  - 8 bi-encoders: USE, MiniLM, MPNet, E5-small, E5-base, BGE-small, BGE-base, Instructor
  - 2 cross-encoders: BERT-CE, RoBERTa-CE
  - 2 encoder-decoder: T5, FLAN-T5
  - 5 LLMs: Llama 3.2, Mistral 7B, Phi-3, Gemma, Qwen

#### 2. **Architectural Paradigm Analysis** (ENTIRELY NEW)
- Bi-encoders vs cross-encoders vs encoder-decoder
- Trade-off analysis: accuracy vs computational cost
- 48× speedup quantification

#### 3. **LLM Evaluation** (ENTIRELY NEW - 2024 models)
- Zero-shot and few-shot prompting
- Comparison of instruction-tuned LLMs
- Finding: LLMs achieve only 76-91% (far below supervised baseline)

#### 4. **Scalability Analysis** (ENTIRELY NEW)
- Projection to 17M texts
- Bi-encoders: 39 days (feasible)
- Cross-encoders: 4.7 years (infeasible)
- Critical for corpus-scale applications

#### 5. **STS Benchmark Comparison** (ENTIRELY NEW)
- Evaluation on STS-B benchmark
- **Key finding**: Inverse correlation between STS performance and our task
- BGE/E5 dominate STS-B but underperform USE on plot similarity
- Challenges assumption that benchmark performance transfers

#### 6. **Why USE Outperforms Modern Models** (ENTIRELY NEW)
- Hypothesis 1: Multi-task pre-training (conversational + translation)
- Hypothesis 2: Deep Averaging Network architecture
- Hypothesis 3: STS benchmarks reward different inductive biases
- Discussion section analyzing this surprising result

#### 7. **Comprehensive Ablation Studies** (ENTIRELY NEW)
- Effect of classifier type (MLP vs RF vs Logistic vs SVM)
- Effect of text length (500 to 10,000 tokens)
- Zero-shot vs few-shot for LLMs

#### 8. **Error Analysis** (NEW)
- Detailed analysis of 3 false negatives
- Error taxonomy

### Different Framing (CL Journal)
- **Focus**: NLP/computational linguistics
- **Goal**: Benchmark for evaluating sentence encoders on compositional semantic tasks
- **Audience**: NLP researchers, ML practitioners
- **Contribution Type**:
  1. Novel evaluation task (plot-level semantic similarity)
  2. Benchmark dataset
  3. Systematic architecture comparison
  4. Challenge to STS benchmark assumptions

### Different Research Questions
| DH2022 | CL Journal |
|--------|------------|
| Can we find Crusoe adaptations? | How do embedding models perform on plot-level similarity? |
| Does USE work for this task? | Why does USE outperform modern models? |
| Can ML learn plot implicitly? | Do STS benchmarks predict performance on compositional tasks? |
| | Which architecture is suitable for corpus-scale analysis? |

---

## Overlap Percentage Estimate

### Content Overlap
- **Dataset description**: ~5% of paper (same dataset, but justified differently)
- **USE baseline method**: ~10% of paper (one model among 15)
- **High accuracy result**: Mentioned but not the focus

### New Content
- **Model comparison**: ~30% (15 models, 3 paradigms)
- **Scalability analysis**: ~10%
- **STS benchmark analysis**: ~10%
- **Discussion of why USE wins**: ~15%
- **Ablation studies**: ~10%
- **Related work (updated 2022-2025)**: ~10%

**Estimate: 80% new content, 20% builds on DH2022**

---

## Publication Ethics Considerations

### Why This IS Publishable

1. **Different Venue Type**
   - DH2022: Conference presentation (short abstract, no peer review for full paper)
   - CL Journal: Full peer-reviewed journal article (~10,000 words)

2. **Different Contribution**
   - DH2022: "Here's a method that works"
   - CL Journal: "Here's a systematic evaluation revealing surprising findings about embedding models"

3. **Substantive New Findings**
   - Inverse correlation between STS benchmarks and plot-similarity
   - USE outperforms 2024 SOTA models (E5, BGE)
   - Architectural choice determines feasibility
   - LLMs fail at this task despite general capabilities

4. **Updated Context**
   - DH2022 was 2022 (pre-LLM era)
   - CL paper includes 2024 models (Llama 3.2, Mistral, Gemma, etc.)
   - Updated related work with 2022-2025 research

5. **Precedent**
   - Common to expand conference presentations into journal articles
   - Standard practice: acknowledge prior presentation, emphasize new contributions

### What We MUST Do

✅ **Acknowledge DH2022 presentation** in the paper:
```latex
This work builds on our preliminary findings presented at DH2022 [Glass2022],
which demonstrated that Universal Sentence Encoder could detect Robinson Crusoe
adaptations with high accuracy. The present paper significantly extends that work
with systematic evaluation of 15 embedding models across three architectural
paradigms, analysis of why USE outperforms modern models despite inferior STS
benchmark performance, and demonstration of architectural constraints for
corpus-scale applications.
```

✅ **Emphasize NEW contributions** in abstract and introduction

✅ **Different title** that signals benchmark/evaluation focus

---

## Recommended Changes

### 1. Title Change (Current vs Suggested)

**Current Title:**
"Plot-Level Semantic Similarity Detection in Literary Texts: A Benchmark for Long-Document Sentence Encoders"

**Issues:** Doesn't distinguish from DH2022 clearly enough

**Suggested Alternatives:**

**Option A (Emphasize systematic comparison):**
"Evaluating Sentence Encoders for Plot-Level Semantic Similarity: Why Universal Sentence Encoder Outperforms Modern Models"

**Option B (Emphasize benchmark contribution):**
"A Plot-Level Semantic Similarity Benchmark: Systematic Evaluation of Bi-Encoders, Cross-Encoders, and LLMs on Long Documents"

**Option C (Emphasize surprising finding):**
"Beyond Sentence-Level Similarity: Why Universal Sentence Encoder Outperforms Modern Embeddings on Plot Detection"

**Option D (Emphasize architectural analysis):**
"Architectural Paradigms for Plot-Level Semantic Similarity: A Systematic Evaluation of 15 Embedding Models"

**Recommendation: Option B or D** - clearly signals this is a benchmark/evaluation paper, not an application paper

### 2. Abstract Changes

**Add to beginning:**
"Recent embedding models (E5, BGE) dominate sentence-level similarity benchmarks, but their performance on document-level compositional semantic tasks remains unexplored."

**Add acknowledgment** (in Introduction, not abstract):
"This work extends our preliminary DH2022 study [Glass2022] which demonstrated proof-of-concept for USE-based adaptation detection."

### 3. Introduction Changes

**Current opening** (para 1):
"Sentence embedding models—including Universal Sentence Encoder..."

**Suggested opening:**
"Sentence embedding models are typically evaluated on sentence-level semantic textual similarity (STS) benchmarks. However, whether performance on these benchmarks transfers to document-level compositional semantic tasks—where texts share high-level structure but differ in surface features—remains an open question. We introduce plot-level semantic similarity detection as a challenging evaluation task that tests this transfer assumption."

**Add new paragraph 2:**
"Our work builds on preliminary findings presented at DH2022 [Glass2022], which demonstrated that Universal Sentence Encoder (USE) could detect Robinson Crusoe adaptations with 99% accuracy. The present paper significantly extends that work by: (1) systematically evaluating 15 models across three architectural paradigms, (2) demonstrating that modern embeddings (E5, BGE) underperform USE despite dominating STS benchmarks, (3) quantifying architectural constraints for corpus-scale analysis, and (4) evaluating recent LLMs (Llama, Mistral, etc.) on this task."

### 4. Related Work Addition

**Add subsection: "Prior Work on This Dataset"**

```latex
\paragraph{DH2022 Preliminary Study}
Glass (2022) introduced the Robinson Crusoe adaptation detection task,
demonstrating that USE embeddings outperformed TF-IDF and BERT for this
application. That study compared 3 models and achieved high accuracy, but
did not systematically evaluate architectural paradigms, compare to modern
embeddings (E5, BGE), or analyze why USE succeeds. The present work provides
comprehensive evaluation and theoretical analysis of these questions.
```

### 5. Contributions Section Rewrite

**Current:**
"We make four primary contributions..."

**Suggested (emphasize what's NEW):**

```latex
Building on the proof-of-concept study by Glass (2022) at DH2022, we make
four primary contributions:

1. **Systematic Architectural Evaluation**: Comprehensive comparison of 15
   models across three paradigms (bi-encoders, cross-encoders, encoder-decoder),
   revealing that architectural choice determines computational feasibility
   for corpus-scale applications.

2. **STS Benchmark Transfer Analysis**: First demonstration that STS benchmark
   rankings inversely correlate with plot-level similarity performance,
   challenging assumptions about model selection.

3. **LLM Evaluation**: Assessment of recent instruction-tuned LLMs (Llama 3.2,
   Mistral 7B, etc.) showing 76-91% accuracy, far below supervised baselines.

4. **Theoretical Analysis**: Investigation of why USE outperforms modern
   embeddings (E5, BGE) optimized on 2024 benchmarks, with implications for
   embedding model design.
```

---

## Bibliography Addition

Add to references_cl.bib:

```bibtex
@inproceedings{Glass2022,
  author    = {Grant Glass},
  title     = {An Adaptive Methodology: Machine Learning and Literary Adaptation},
  booktitle = {Digital Humanities 2022: Conference Abstracts},
  year      = {2022},
  address   = {Tokyo, Japan},
  publisher = {DH2022 Local Organizing Committee},
  note      = {Proof-of-concept study demonstrating USE for Crusoe adaptation detection}
}
```

---

## Summary: Is This Publishable?

### YES ✅

**Reasoning:**
1. **80% new content**: 15 models vs 3, architectural analysis, LLMs, scalability, STS comparison
2. **Different framing**: Benchmark paper vs application paper
3. **Different contribution**: Systematic evaluation revealing surprising findings vs proof-of-concept tool
4. **Updated context**: Includes 2024 models and research
5. **Different audience**: NLP/ML community vs DH community
6. **Proper acknowledgment**: Will cite DH2022 and clarify what's new

**This is standard practice**: Expanding conference presentations into journal articles is common and ethical when:
- Substantive new work is added (✓ 80% new)
- Prior presentation is acknowledged (✓ will add)
- New contributions are clearly stated (✓ will emphasize)

### Recommended Actions (Priority Order)

1. **Change title** to emphasize benchmark/evaluation focus (Option B or D)
2. **Add acknowledgment** of DH2022 in Introduction
3. **Rewrite contributions** section to emphasize what's NEW
4. **Add "Prior Work on This Dataset"** subsection to Related Work
5. **Add Glass2022 citation** to bibliography
6. **Update abstract** to position as investigating transfer from STS benchmarks

---

## Final Recommendation

**Proceed with submission** after making the changes above. The paper is substantively different and makes novel contributions that significantly advance the field beyond the DH2022 presentation.

**Key message to emphasize:**
"We use the same dataset as our DH2022 proof-of-concept, but ask fundamentally different questions: not 'can we detect adaptations?' but 'which embedding architectures perform best on plot-level similarity, and why does an older model (USE) outperform modern SOTA models (E5, BGE)?'"
