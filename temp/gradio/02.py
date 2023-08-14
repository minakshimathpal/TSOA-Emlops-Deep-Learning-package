import gradio as gr
import torch
import requests

from PIL import Image
from torchvision import transforms


def test_fun(name:str,check:bool,slider:int):
    return f"{name=},{check=},{slider=}"



gr.Interface(fn=test_fun, inputs=[gr.Text(),gr.Checkbox(),gr.Slider(0,100,step=5)], outputs=[gr.Text()]).launch(share=True)
