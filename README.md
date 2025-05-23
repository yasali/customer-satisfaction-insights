# Customer Feedback Insights Dashboard

An interactive dashboard that uses a local LLM (Flan-T5) and Gradio to provide meaningful insights from customer satisfaction data.

## Features

- Key drivers of satisfaction using correlation analysis
- Interactive pie chart of satisfaction levels
- AI-generated text insights using Flan-T5
- Fully local — data never leaves your machine

## Tech Stack

| Tool / Library    | Purpose                               |
|-------------------|----------------------------------------|
| pandas            | Data loading and manipulation          |
| plotly            | Visualizations                         |
| gradio            | Web-based UI                           |
| transformers      | Language model pipeline                |
| torch             | Runs the model backend                 |
| python-dotenv     | Environment variables support          |

## Installation

Install all required packages:

`pip install pandas plotly gradio transformers torch python-dotenv`

(Optional) If you want GPU acceleration:

`pip install accelerate`


## Usage

1. Replace the placeholder with the correct path to your CSV file in `dashboard.py`:

`path = "your/path/here"`


2. Run the app:

`python dashboard.py`


3. Gradio will open in your browser (typically at http://127.0.0.1:7860).

## Dataset Requirements

The input CSV must include the following columns:

- ProductQuality
- ServiceQuality
- PurchaseFrequency
- SatisfactionScore
- LoyaltyLevel
- Country

Each row represents an individual customer response.

## Model Info

The dashboard uses the `google/flan-t5-base` model from Hugging Face, which performs summarization locally via the `transformers` library. It supports both CPU and GPU execution.

## File Structure

dashboard.py # Main app logic
customer_feedback_satisfaction.csv # Input dataset
.env # Optional env variables
README.md # Project documentation


## Screenshots

![Screenshot 2025-05-24 at 00 28 39](https://github.com/user-attachments/assets/9eb2392f-75c7-4ec0-a52e-4db28ed5c09a)

## Acknowledgements

- Hugging Face for Flan-T5
- Gradio for rapid UI development
- Plotly for clean visualizations

## License

This project is open-sourced under the MIT License.
