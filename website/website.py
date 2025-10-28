import gradio as gr
from model.model import recommend_wine, food_items, food_category, cuisine

from numba.scripts.generate_lower_listing import description

iface = gr.Interface(
    fn = recommend_wine,
    inputs  = [gr.Dropdown(choices=food_items, label= "Select a food"),
               gr.Dropdown(choices=food_category, label ="Food category"),
               gr.Dropdown(choices= cuisine, label = "Select a cuisine")],
    outputs = gr.Markdown(label="Recommended Wine"),
    title="Food and Wine Pairing Predictor",
    description = "Select a food and get a wine recommendation"
)

iface.launch(share=True)

