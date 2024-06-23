# PDF-Chat-Assistant

## Description

**PDF-Chat-Assistant** is an advanced AI-powered tool that transforms how you interact with PDF documents. Leveraging the capabilities of OpenAI's GPT-3.5-turbo and state-of-the-art vector stores, this tool allows you to upload PDF files and ask questions to extract valuable information quickly and efficiently. Whether you're a researcher, student, or professional, **PDF-Chat-Assistant** makes it easier to find the answers you need from your documents.

## Features

- **Seamless PDF Upload:** Easily upload multiple PDF files for processing and querying.
- **AI-Powered Question Answering:** Get detailed answers to your questions from the content of the PDFs using advanced language models.
- **Efficient Text Extraction:** Extract text from PDFs and split it into manageable chunks for better processing.
- **Vector Store Integration:** Utilize FAISS vector stores for efficient storage and retrieval of document chunks.
- **Interactive User Interface:** Engage with the tool through a user-friendly Streamlit interface.
- **Context-Aware Responses:** The assistant provides detailed answers based on the context of the documents, ensuring accuracy.

## Use Cases

- **Research:** Extract and analyze information from academic papers and research documents.
- **Education:** Quickly find answers from textbooks and study materials.
- **Business:** Retrieve data from reports, whitepapers, and business documents.
- **Legal:** Extract relevant information from legal documents and contracts.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/PDF-Chat-Assistant.git
    ```

2. **Navigate to the project directory:**
   ```bash
   cd PDF-Chat-Assistant
    ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    ```bash
    cd PDF-Chat-Assistant
    ```
5. **Set up environment variables:**
    ```bash
        ANTHROPIC_API_KEY=your_anthropic_api_key
        OPENAI_API_KEY=your_openai_api_key
    ```
6. **Run the Application:**
    ```bash
        streamlit run app.py
    ```

## Usage
**Upload PDFs:**
Use the intuitive interface to upload your PDF files.


**Process Documents:**
Click on "Submit & Process" to read and process the PDFs, extracting and chunking the text for efficient querying.


**Ask Questions:**
Enter your questions in the provided input field, and the assistant will retrieve and display the most relevant answers based on the document content.


**Get Answers:**
Receive detailed and context-aware responses to your queries, helping you find the information you need quickly.
