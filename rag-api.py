import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

def set_rag():
        os.environ["GOOGLE_API_KEY"] = ""
        if not os.environ.get("GOOGLE_API_KEY"):
                print("ERRO: A chave de API do Google não foi configurada.")
                return None

        # Fonte de dados
        print("[1/5] - Carregando documento web...")
        start_time = time.time()
        texto_exemplo = """
                O ciclo da água, também conhecido como ciclo hidrológico, é o processo contínuo de movimento da água na Terra. Este ciclo é fundamental para a existência da vida no planeta, pois garante a distribuição de água potável, regula o clima e modela a paisagem. A água existe em três estados principais durante o ciclo: líquido, gasoso (vapor) e sólido (gelo).

                A primeira grande fase é a evaporação. O calor do sol aquece a água de rios, lagos e oceanos, transformando-a em vapor. Esse vapor de água sobe para a atmosfera. As plantas também liberam vapor de água através de um processo chamado transpiração, contribuindo para a umidade do ar.

                Quando o vapor de água atinge altitudes mais elevadas na atmosfera, ele esfria e se transforma de volta em pequenas gotículas de água líquida ou cristais de gelo. Esse processo é chamado de condensação. Essas gotículas se agrupam e formam as nuvens que vemos no céu.

                Finalmente, ocorre a precipitação. Quando as gotículas de água nas nuvens se tornam grandes e pesadas o suficiente, elas caem de volta para a superfície da Terra na forma de chuva, neve ou granizo. A água que cai é coletada em rios, lagos e oceanos, ou se infiltra no solo, recomeçando o ciclo hidrológico.
                """

        docs = [Document(page_content=texto_exemplo)]
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
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'} # Use 'cuda' se tiver uma GPU compatível
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        print(f"         {time.time()-start_time}s.")

        #Config llm e prompt
        print("[4/5] - Configurando llm...")
        start_time = time.time()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
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
        print ("\n--- PERGUNTA ---")
        pergunta = input("Digite sua pergunta, ou 'sair' para terminar\n")

        if pergunta.lower() == "sair":
                print("Encerrando...")
                exit()
 # Invoca a chain e mede o tempo de resposta
        start_time = time.time()
        response = rag_chain.invoke({"input": pergunta})
        end_time = time.time()

        # Imprime a resposta
        print("\n--- RESPOSTA ---")
        print(response["answer"])
        print(f"\n(Resposta gerada em {end_time - start_time:.2f} segundos)")

        
        print("\n--- CONTEXTO UTILIZADO ---")
        for i, doc in enumerate(response["context"]):
                print(f"\n[Trecho {i+1}]")
                print(doc.page_content)