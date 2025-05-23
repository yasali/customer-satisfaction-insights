import pandas as pd
import dotenv
import gradio as gr
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

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
gen_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer
)

# Generate analytic insights programmatically
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


# Visualization function
def satisfaction_pie():
    pie_data = df['SatisfactionLevel'].value_counts().reset_index()
    pie_data.columns = ['Satisfaction', 'Count']
    return px.pie(pie_data, names='Satisfaction', values='Count', title='Overall Satisfaction')

# Build Gradio dashboard
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# 📊 Customer Feedback Insights (Local LLM)")
    with gr.Row():
        chart = gr.Plot(label="Satisfaction Breakdown")
        insights = gr.HTML(label="AI-Generated Insights")
    gen_btn = gr.Button("Generate Insights")

    # Display pie on startup
    dashboard.load(fn=satisfaction_pie, inputs=[], outputs=[chart])

    # Bind button click
    gen_btn.click(fn=generate_insight, inputs=[], outputs=[insights])

if __name__ == "__main__":
    dashboard.launch()
