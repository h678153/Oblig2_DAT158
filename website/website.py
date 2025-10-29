import gradio as gr
from model.model import recommend_wine, food_items

iface = gr.Interface(
    fn = recommend_wine,
    inputs  = gr.Dropdown(choices=food_items, label= "Select a food", allow_custom_value=True),
    outputs = gr.Markdown(label="Recommended Wine"),
    title="Food and Wine Pairing Predictor",
    description = "Select a food and get a wine recommendation"
)

iface.launch(share=True)
