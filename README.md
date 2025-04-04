# Personal-RAG-System
A personal Retrieval-Augmented Generation (RAG) system built with the DeepSeek-R1 model, Langchain, and Streamlit. This system allows you to upload PDFs to the Pinecone database for document retrieval and leverage DeepSeek-R1 reasoning power for high-quality response.

# Pre-requisites
Install Ollama on your local machine from the [official website](https://ollama.com/). And then pull the Deepseek model:

```bash
ollama pull deepseek-r1:1.5b
```

This project utilizes a 1.5B parameter model by default. You can modify the model selection in the model_setting.py file:

```bash
model='deepseek-r1:1.5b'
```

Update the vector_size parameter in model_setting.py to match the selected model's embedding dimension.
By default, it is set to 1536 for a model with 1.5B parameters.

```bash
vector_size=1536
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

# Run the application

```bash
streamlit run app.py
```

