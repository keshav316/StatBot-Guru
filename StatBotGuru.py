from langchain_ollama import ChatOllama
import gradio as gr
from langchain.memory import ConversationBufferMemory


# 🧠 Connect to your local LLM
llm = ChatOllama(model="StatBotGuru", temperature=0.3)
# 💬 Define a generator function for streaming
memory= ConversationBufferMemory()

def get_stat_answer(question):
    previous_chat = memory.buffer
    prompt = previous_chat + f"\nHuman: {question}\nAI:"
    partial = ""
    for chunk in llm.stream(prompt):
        partial += chunk.content
        yield chunk.content
    memory.save_context({"input": question}, {"output": partial})
# 🖼️ Build Gradio interface
interface = gr.Interface(
    fn=get_stat_answer,
    inputs=gr.Textbox(
        label="🧮 Ask StatBotGuru",
        placeholder="Ask any question about statistics, probability, or math used in ML/AI...",
        
    ),
    outputs=gr.Textbox(label="📘 Response"),
    title="📊 StatBotGuru",
    description="Your AI tutor for statistics, probability, and math used in ML/AI. Runs locally via LangChain + Ollama.",
)

# 🚀 Enable streaming and launch
interface.queue()  # Required for streaming
interface.launch()
