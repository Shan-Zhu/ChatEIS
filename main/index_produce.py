import os
import openai
import llama_index as li
from langchain import OpenAI

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = 'OpenAI_API_key'

def generate_index(doc_dir, persist_dir):
    # Load documents from directory
    documents = li.SimpleDirectoryReader(doc_dir).load_data()

    # Initialize LangChain and LLMPredictor
    langchain = OpenAI(temperature=0, model_name="text-embedding-ada-002")
    llm_predictor = li.LLMPredictor(llm=langchain)

    # Initialize PromptHelper and ServiceContext
    prompt_helper = li.PromptHelper(max_input_size=4096, num_output=256, max_chunk_overlap=20)
    service_context = li.ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Generate index using GPTVectorStoreIndex
    index = li.GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    # Persist index to storage
    index.storage_context.persist(persist_dir=persist_dir)

    return index

# Example usage
index = generate_index(doc_dir='EIS_paper', persist_dir='index_storage')
print('OK')
