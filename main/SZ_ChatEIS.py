

import tkinter as tk
import os
import gradio as gr
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, \
    StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from langchain.chat_models import ChatOpenAI

class ChatEIS:
    def __init__(self, documents_dir, persist_dir):
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))  #temperature=0,
        self.max_input_size = 4096
        self.num_output = 256
        self.max_chunk_overlap = 20
        self.prompt_helper = PromptHelper(self.max_input_size, self.num_output, self.max_chunk_overlap)
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor,
                                                            prompt_helper=self.prompt_helper)
        self.index = None
        self.query_engine = None

    def load_index(self):
        if os.path.exists(self.persist_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            documents = SimpleDirectoryReader(self.documents_dir).load_data()
            self.index = GPTVectorStoreIndex.from_documents(documents, service_context=self.service_context)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        self.query_engine = self.index.as_query_engine()

    def query(self, text):
        if self.query_engine is None:
            self.load_index()
        response = self.query_engine.query(text)
        return response

class ChatWindow:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")

        # Set default API key
        self.default_api_key = "OPENAI_API_KEY"
        os.environ["OPENAI_API_KEY"] = self.default_api_key

        # Create button to change API key
        self.api_button = tk.Button(master, text="Change API Key", command=self.change_api)
        self.api_button.pack()

        # Create text entry box
        self.text_label = tk.Label(master, text="Enter your question:")
        self.text_label.pack()
        self.text_entry = tk.Entry(master, width=50)
        self.text_entry.pack(fill=tk.BOTH, expand=True)
        self.text_entry.insert(tk.END, "We are testing the impedance of a 'lithium-ion battery' and the equivalent circuit obtained is 'p(R_0,CPE_0)-p(R_1,CPE_1)-p(R_2,CPE_2)'")

        # Create button to submit question
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_question)
        self.submit_button.pack()

        # Create text box to display response
        self.response_label = tk.Label(master, text="Response:")
        self.response_label.pack()
        self.response_text = tk.Text(master, height=10)
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # Create text box to display history
        self.history_label = tk.Label(master, text="History:")
        self.history_label.pack()
        self.history_text = tk.Text(master, height=10)
        self.history_text.pack(fill=tk.BOTH, expand=True)

    def change_api(self):
        # Open API window to change API key
        api_window = tk.Toplevel(self.master)
        api_window.geometry("400x100")
        api_window.resizable(False, False)
        api_window.lift()
        api_window.attributes('-topmost', True)
        api_window.focus_force()

        def submit_api_key():
            # Get API key from entry box
            api_key = api_entry.get()

            # Set OpenAI API key
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = self.default_api_key

            # Close API window
            api_window.destroy()

        # Create API key entry box and change button
        api_label = tk.Label(api_window, text="Enter OpenAI API key:")
        api_label.pack()
        api_entry = tk.Entry(api_window, width=50)
        api_entry.pack(fill=tk.BOTH, expand=True)
        api_entry.insert(tk.END, self.default_api_key)
        api_button = tk.Button(api_window, text="Change API Key", command=submit_api_key)
        api_button.pack()

    def submit_question(self):
        question = self.text_entry.get()
        question = "You are now playing the role of an electrochemistry expert. Please analyze the following information and provide your expert insights." + question

        chatbot = ChatEIS('eis_doc', 'index_storage')
        response = chatbot.query(question)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, response)

        # Add question and response to history
        history_question = question.replace("You are now playing the role of an electrochemistry expert. Please analyze the following information and provide your expert insights.", "")
        history_text = f"Question: {history_question}\nResponse: {response}\n\n"
        self.history_text.insert(tk.END, history_text)


root = tk.Tk()
chat_window = ChatWindow(root)
root.mainloop()
