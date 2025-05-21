# Pipeline to process and extract ordered items construction tenders and extracted products to an existing database

## This project leverages LLMs to automate a manual, time-intensive process

1. Tenders are read in using pdfplumber.
2. Tenders are very long and need to be broken up, so text is processed and chunked in a way that keeps all relevant context for
   item ordering together.
3. Text chunks are sent to GPT-4o-mini using the OpenAI API.
4. LLM output is converted to a Pandas dataframe.
5. Vector embeddings of extracted items and database items are created using OpenAIs embeddings3-small model.
6. A FAISS index is created and extracted items are matched to database items using cosine similarity.
7. The top 5 matches are sent back to GPT-4o-mini to verify match quality.
8. If the LLM confirms there is a true match:
   
     -the match is stored in the respective dataframe column
10. If no suitable match is found:
    
      -the respective position is left empty in the dataframe
12. A dataframe containing the extracted items, quantities, units, and database matches is the final output.
