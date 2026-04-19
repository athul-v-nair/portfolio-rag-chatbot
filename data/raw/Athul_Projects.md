# Projects History

<p>Includes the projects which are tried for learning and and real ones. Acending order - latest one is first and oldest one last</p>

## Project 1: Vectorless RAG - Interview QA

-  Built a RAG system without vector embeddings, using hierarchical document reasoning to improve context retention and reduce fragmented retrieval.
- Implemented structure-first indexing (Sections → Subsections → Paragraphs) with PageIndex, enabling precise navigation of long documents (60+ pages).
- Replaced similarity search with LLM-driven reasoning, improving relevance for complex, multi-step queries.
- Achieved more context-complete and cited answers compared to traditional chunk-based RAG approaches.
- Tech Stack: Python, PageIndex, Groq (LLaMA 3/3.1).
- Github Repository: https://github.com/athul-v-nair/vectorless-interview-qa-prep

## Project 2: Finetuning DistilBERT model to predict Movie Genres

-  Fine-tuned distilbert-base-uncased on a 8,000-sample movie dataset to classify titles and synopses into 10 genres, improving validation accuracy from a random 10% baseline to ~39% within 3 epochs on a CPU-constrained environment.
- Diagnosed and resolved common fine-tuning failure modes including overfitting, learning rate instability, and class imbalance through iterative experimentation.
- Implemented full training pipeline using Hugging Face Trainer API with warmup scheduling, cosine learning rate decay, early stopping, and dropout regularization to improve generalization.
- Explored model selection trade-offs between MiniLM, DistilBERT, and BERT-base, observing real-world impact of architecture choices on convergence and accuracy.
- Analysed training curves to identify overfitting (train loss 0.81 vs val loss 2.26) and applied corrective techniques like including a bigger max_length tuning for longer synopses.
- Tech Stack: Hugging Face Transformers, PyTorch, Datasets, scikit-learn.

## Project 3: Portfolio AI Assistant

-  Built a Retrieval-Augmented Generation (RAG)chatbot for a portfolio website, enabling context-aware
question answering over personal and project data.
- Designed and implemented end-to-end pipelines for ingestion (parsing, chunking, embedding, vector
storage), retrieval(query embedding, chat history) and generation.
- Implemented session-based conversational memory, allowing responses to leverage historical chat con
text for improved coherence.
- Developed scalable backend API using FastAPI and integrated seamlessly with frontend.
- Evaluated system using retrieval and generation metrics, achieving Recall@3: 1.0, MRR: 1.0, with Con
text Precision: 0.50
- Tech Stack: LangChain, ChromaDB, FastAPI, Docker, Git
- Github Repository: https://github.com/athul-v-nair/portfolio-rag-chatbot

## Project 4: Text Generation Transformer (From Scratch + MLOps)

- Implemented a decoder-only Transformer architecture from scratch in PyTorch for autoregressive
text generation using the WikiText-2 dataset.
- Designed core transformer components including multi-head causal self-attention, positional embed
dings, masking, and feed-forward networks.
- Built a modular architecture consisting of token embeddings, stacked transformer blocks, residual connec
tions, and final projection layers.
- Implemented training pipeline with checkpointing, metric logging, and perplexity evaluation.
- Developed multiple text generation strategies including greedy decoding, temperature sampling, top-k,
top-p, and hybrid sampling.
- Created an API inference service using FastAPI and containerized the pipeline using Docker for repro
ducible deployment.
- Github Repository: https://github.com/athul-v-nair/text-generation-mlops

## Project 5: Immersify – AI Powered Immersive Reading Experience

- Immersive Reading Experience for Kids & Visually Impaired
- Developed an AI-powered accessibility application that generates immersive sound experiences during
read-aloud sessions.
- Implemented real-time identification of sound producing words to trigger contextual audio effects
based on detected words (e.g., rain sounds when ”rain” is read).
- Developed an AI-powered immersive reading application that received positive feedback from judges and
participants during a hackathon evaluation.
- Built the backend using AWS Bedrock for model integration and AWS Transcribe for speech-to-text
processing, while storing application data in AWS S3.
- Experimented with AWS OpenSearch to accelerate semantic search and retrieval of contextual audio
triggers; evaluated performance gains against infrastructure cost to guide architectural decisions.

## Project 6: Contrack - Contract Management System

- Enterprise contract management and tracking system
- Built a full-stack SaaS platform to manage enterprise contracts and track revenue projections.
- Implemented CRUD operations, role-based access control, and SSO authentication.
- Designed analytics dashboards for monitoring contract revenue and business insights.
- Tech Stack: React.js, Laravel, PHP, MySQL, REST APIs.
- Github Repository: https://github.com/PranavSR04/ConTrack---RJS-Experion