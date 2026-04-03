from flask import Flask, render_template, request, session
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from src.helper import download_hugging_face_embeddings
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

print("KEY BEING USED:", os.getenv("GEMINI_API_KEY")[:10])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

app = Flask(__name__)
app.secret_key = "medical-chatbot-secret"

print("🔄 Loading embeddings model...")
embeddings = download_hugging_face_embeddings()
print("✅ Embeddings loaded")

index_name = "medical-chatbot"
print("🔄 Connecting to Pinecone...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
print("✅ Pinecone connected")

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

print("🔄 Initializing Gemini model...")
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
print("✅ Gemini ready")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("🚀 Application initialized!")


@app.route("/")
def index():
    session.clear()
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "❌ No input provided"

    # Load history from session
    if "chat_history" not in session:
        session["chat_history"] = []

    # Build LangChain message history
    history = []
    for turn in session["chat_history"]:
        history.append(HumanMessage(content=turn["human"]))
        history.append(AIMessage(content=turn["ai"]))

    try:
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": history
        })
        answer = response.get("answer", "No response generated")

        # Save to session
        session["chat_history"].append({"human": msg, "ai": answer})
        session.modified = True

        print("User:", msg)
        print("Bot:", answer)
        return answer

    except Exception as e:
        print("🔥 Error:", e)
        return "⚠️ Something went wrong"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)