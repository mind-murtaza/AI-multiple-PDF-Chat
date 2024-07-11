import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from frontend import css, bot_template, user_template

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def generate_chunk_summary(chunk, model):
    """
    Generate a summary for a given text chunk using a language model.
    
    Parameters:
    - chunk: The text chunk to summarize.
    - model: The language model instance to use for summarization.
    
    Returns:
    - A summary of the text chunk.
    """
    prompt = f"Summarize the following text:\n\n{chunk}"
    # response = model.call_as_llm(prompt)  # Assuming call_as_llm is synchronous OpenAI API call
    response = model.invoke(prompt) # Azure OpenAI

    return response


def summarize_text(text):
    """
    Iterate over text chunks, generate summaries, and yield them one by one.
    
    Parameters:
    - text: The full text to be summarized.
    
    Yields:
    - A summary for each chunk of text.
    """
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Initialize the language model OpenAI
    llm = AzureOpenAI(deployment_name="gpt-35-turbo") # Azure OpenAI
    chunks = get_text_chunks(text)  # Get text chunks
    
    for chunk in chunks:
        summary = generate_chunk_summary(chunk, llm)
        yield summary

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo") # OpenAI
    llm = llm = AzureOpenAI(deployment_name="gpt-35-turbo") # Azure OpenAI
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and generate response
def handle_userinput(user_question):
    if not callable(st.session_state.conversation):
        st.warning("Please upload and process the PDF documents first.")
        return

    summary_keywords = ["summarize", "summary", "summarise", "synopsis", "overview"]

    if any(keyword in user_question.lower() for keyword in summary_keywords):
        raw_text = st.session_state.raw_text
        # Start generating summaries immediately
        summary_generator = summarize_text(raw_text)
        st.write("### Summary")
        next(summary_generator)  # Consume the first summary
        while True:
            try:
                summary = next(summary_generator)
                st.write(summary)
            except StopIteration:
                break  # No more summaries to generate/display
    else:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Reverse the chat history to display the latest message at the top
        reversed_chat_history = list(reversed(st.session_state.chat_history))

        st.write('<div class="chat-history">', unsafe_allow_html=True)
        for i, message in enumerate(reversed_chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        st.write('</div>', unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)  # Add the CSS here

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.raw_text = raw_text

if __name__ == '__main__':
    main()