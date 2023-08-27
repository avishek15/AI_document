This repository is for querying your document
and be able to discuss crucial insights from the document
through OpenAI's chatbot models.


I am utilising Langchain, SentenceTransformer and OpenAI API.

Done:
- [x] Basic Chatbot without memory
- [x] Document Uploading interface
- [x] Handled the case for only pdf documents to be uploaded.
- [x] Document storage into Chroma DB
- [x] Connect bot with VectorDB
- [x] Context embeddings should return good quality matches
- [x] Force the bot to answer from context

To Do:
- [ ] Unified Interface
- [ ] Test for the case for the question for which the book has no information 
- [ ] Handling no context (If no file is uploaded)
- [ ] Chatbot Memory
- [ ] Turn into WebApp (Flask maybe?)


### <u>Running the example</u>
- Install the necessary packages (Requirements.txt)
- Run chatbot.py


