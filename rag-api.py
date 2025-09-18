import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

def set_rag():
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCzUscBcxKVwSRqZdwXBZ8DlrEFKPPKcfE"
        if not os.environ.get("GOOGLE_API_KEY"):
                print("ERRO: A chave de API do Google não foi configurada.")
                return None

        # Fonte de dados
        print("[1/5] - Carregando documento web...")
        start_time = time.time()
        loader = WebBaseLoader("https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial", requests_kwargs={"verify": False})
        docs = loader.load()
        print(f"Documento carregado com {len(docs[0].page_content)} caracteres.")
        print(f"         {time.time()-start_time}s.")

        #Chunking
        print("[2/5] - Realizando chunking do documento...")
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"         {time.time()-start_time}s.")

        #Criar vetores e salvar no bd
        print("[3/5] - Criando banco de dados vetorial (Pode demorar um pouco)...")
        start_time = time.time()
        vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        retriever = vectorstore.as_retriever()
        print(f"         {time.time()-start_time}s.")

        #Config llm e prompt
        print("[4/5] - Configurando llm...")
        start_time = time.time()
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        prompt = ChatPromptTemplate.from_template("""
        Responda à pergunta do usuário baseando-se somente no contexto fornecido.
        Se a informação não estiver no contexto, diga "Não tenho informações sobre isso no meu contexto".

        <context>
        {context}
        </context>

        Pergunta: {input}
        """)
        print(f"         {time.time()-start_time}s.")

        #conecta a llm com o prompt criado
        print("[5/5] - Construindo RAG llm...")
        start_time = time.time()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print(f"         {time.time()-start_time}s.")
        return retrieval_chain

if __name__ == "__main__":
        rag_chain = set_rag()
        pergunta = input("Digite sua pergunta, ou 'sair' para terminar")

        if pergunta_usuario.lower() == "sair":
                print("Encerrando...")


 # Invoca a chain e mede o tempo de resposta
        start_time = time.time()
        response = rag_chain.invoke({"input": pergunta_usuario})
        end_time = time.time()

        # Imprime a resposta
        print("\n--- RESPOSTA ---")
        print(response["answer"])
        print(f"\n(Resposta gerada em {end_time - start_time:.2f} segundos)")

        
        print("\n--- CONTEXTO UTILIZADO ---")
        for i, doc in enumerate(response["context"]):
                print(f"\n[Trecho {i+1}]")
                print(doc.page_content)