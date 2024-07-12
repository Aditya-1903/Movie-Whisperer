# Movie-Whisperer

A simple Question-Answering (Q&A) bot that uses Retrieval Augmented Generation (RAG) to answer questions regarding the highest grossing Hollywood and Mollywood movies of 2024. The bot can answer questions covering everything from plot summaries and cast details to release dates and reviews. Text data was scraped from Wikipedia and a vector index was generated from it. The Large Language Model (LLM) uses the top 5 documents retrieved from this vector index to generate the final response. 
  
  Embedding Model: BAAI/bge-small-en-v1.5  
  LLM : Mixtral-8x-7b
