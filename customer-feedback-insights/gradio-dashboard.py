import pandas as pd
import dotenv
import gradio as gr
import plotly.express as px
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

dotenv.load_dotenv()

# Load dataset
df = pd.read_csv("customer-feedbacks.csv")

# Preprocess: bucket satisfaction levels
df['SatisfactionLevel'] = pd.cut(
    df['SatisfactionScore'],
    bins=[-1, 49.99, 74.99, 100],
    labels=['Unsatisfied', 'Neutral', 'Satisfied']
)

# Local LLM setup (Flan-T5 for summarization)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
gen_pipeline = hf_pipeline(
    "summarization", 
    model=model, 
    tokenizer=tokenizer
)

# RAG generator pipeline (smaller model for faster response)
generator_model = "t5-small"
llm_pipeline = hf_pipeline("text2text-generation", model=generator_model)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Prompt template (updated to match the new RAG format)
prompt = PromptTemplate.from_template(
    "Use the context to answer the question.\n\nContext: {context}\n\nQuestion: {input}"
)

# Output parser
output_parser = StrOutputParser()

# Create vectorstore with user feedback data
documents = [Document(page_content=desc, metadata={}) for desc in df['tagged_description'].dropna()]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)

# Setup retriever
retriever = vector_store.as_retriever()

# New Retrieval Chain using LCEL (LangChain Expression Language)
retrieval_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# Functions
def summarize_ticket(ticket):
    return retrieval_chain.invoke(ticket)

def generate_insight():
    # Correlation with satisfaction
    corr = df[['ProductQuality', 'ServiceQuality', 'PurchaseFrequency', 'SatisfactionScore']].corr()['SatisfactionScore'].drop('SatisfactionScore')
    corr_sorted = corr.abs().sort_values(ascending=False)
    corr_lines = "".join(f"<li><b>{feat}</b>: {corr[feat]:+.2f}</li>" for feat in corr_sorted.index)

    # Loyalty levels
    loyalty_avg = df.groupby('LoyaltyLevel')['SatisfactionScore'].mean().sort_values(ascending=False)
    loyalty_lines = "".join(f"<li><b>{lvl}</b>: {loyalty_avg[lvl]:.1f}</li>" for lvl in loyalty_avg.index)

    # Top & bottom countries
    country_avg = df.groupby('Country')['SatisfactionScore'].mean()
    top_country = country_avg.idxmax()
    bot_country = country_avg.idxmin()
    top_line = f"<p>🌟 <b>Highest satisfaction:</b> {top_country} ({country_avg[top_country]:.1f})</p>"
    bot_line = f"<p>⚠️ <b>Lowest satisfaction:</b> {bot_country} ({country_avg[bot_country]:.1f})</p>"

    return f"""
        <div style='font-family:sans-serif; line-height:1.6;'>
            <h3>🔍 Key Drivers of Satisfaction</h3>
            <ul>{corr_lines}</ul>
            <h3>🏅 Satisfaction by Loyalty Level</h3>
            <ul>{loyalty_lines}</ul>
            {top_line}
            {bot_line}
        </div>
    """

def satisfaction_pie():
    pie_data = df['SatisfactionLevel'].value_counts().reset_index()
    pie_data.columns = ['Satisfaction', 'Count']
    return px.pie(pie_data, names='Satisfaction', values='Count', title='Overall Satisfaction')

# Gradio dashboard
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# 📊 Customer Feedback Insights (RAG-based Summarization)")
    with gr.Row():
        chart = gr.Plot(label="Satisfaction Breakdown")
        insights = gr.HTML(label="AI-Generated Insights")
        summary = gr.Textbox(label="Support Ticket Summary")
    gen_btn = gr.Button("Generate Insights")
    summarize_btn = gr.Button("Summarize Ticket")
    ticket_input = gr.Textbox(label="Support Ticket")

    dashboard.load(fn=satisfaction_pie, inputs=[], outputs=[chart])
    gen_btn.click(fn=generate_insight, inputs=[], outputs=[insights])
    summarize_btn.click(fn=summarize_ticket, inputs=[ticket_input], outputs=[summary])

if __name__ == "__main__":
    dashboard.launch()
