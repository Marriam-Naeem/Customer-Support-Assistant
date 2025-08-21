# Customer-Support-Assistant

An AI-powered customer support assistant for the **banking domain**, built with Retrieval-Augmented Generation (RAG) and fine-tuned LLMs. The system provides accurate, context-aware answers while ensuring compliance with domain-specific constraints.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Model Fine-tuning](#model-fine-tuning)
- [RAG Pipeline](#rag-pipeline)
- [Security Guardrails](#security-guardrails)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Introduction
Banking customers often struggle to understand complex products and services. Existing chatbots rely heavily on keyword matching and lack domain expertise.  
This project explores how RAG combined with a fine-tuned LLM can improve customer support by delivering accurate, context-aware responses.

## Problem Statement
- Customers face difficulty understanding banking services.
- Existing chatbots are generic and inconsistent.
- High dependency on human agents increases cost and response time.

## Solution
- Domain-specific RAG for banking FAQs and product queries.
- Fine-tuned LLM for context-aware responses.
- Guardrails to ensure safe and compliant answers.
- Modular design for easy integration into banking systems.

## System Architecture


- **Embedding Model:** all-mpnet-base-v2  
- **Vector Store:** ChromaDB  
- **Retriever:** Top-3 passages (cosine similarity)  
- **LLM:** Fine-tuned Phi-1.5 with LoRA  

## Features
- Accurate banking-specific answers.  
- Retrieval-augmented generation pipeline.  
- Fine-tuned LLM with 200+ curated Q&A pairs.  
- Prompt injection and jailbreak protection.  
- Simple user interface for interaction.  

## Technologies Used
- Python  
- LangChain  
- ChromaDB  
- Scikit-learn, Pandas, NumPy  
- Phi-1.5 + LoRA fine-tuning  
- Streamlit / Flask  

## Data Preprocessing
- Custom text cleaning and anonymization.  
- Splitting with Recursive Text Splitter.  
- Embedding generation using all-mpnet-base-v2.  
- Storage and retrieval via ChromaDB.  

## Model Fine-tuning
- **Base Model:** Phi-1.5 (`prd101-wd/phi1_5-bankingqa-merged`)  
- **Fine-tuning Method:** LoRA  
- **Training Data:** 200+ curated banking Q&A pairs  
- Optimized for factual banking-related queries.  

## RAG Pipeline
1. Encode query into embeddings.  
2. Retrieve top-3 relevant documents.  
3. Select context and pass to fine-tuned LLM.  
4. Generate safe and accurate response.  

## Security Guardrails
- Prompt injection prevention.  
- Banned keyword detection.  
- Out-of-domain query rejection.  
- Jailbreak protection.  

## Future Work
- Expand dataset with additional banking use cases.  
- Add multi-lingual support.  
- Integrate with real-time APIs and customer databases.  
- Deploy scalable production version.  

