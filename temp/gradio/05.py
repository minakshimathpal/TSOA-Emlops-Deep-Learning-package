import gradio as gr
import numpy as numpy
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer= GPT2Tokenizer.from_pretrained('gpt2')
model= GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def predict(inp:str)-> str:
    encode_input= tokenizer(inp,return_tensor="pt")
    with torch.no_grad():
        model_out= model.generate(**encoded_input,max_new_tokens=128)
    return tokenizer.decode(model_out[0].tolist())

def flag(inps,outputs):
    print(inps,outputs)

with gr.Blocks(title= "Example with Textbox",theme=gr.themes.Glass()) as demo:
    gr.Markdown(" ## Start typing below and then click **Run** to see the output")

    with gr.Row():
        input_txt= gr.Textbox(placeholder="Input_text",label= "Model Input")
        input_txt.change(lambda inp: print(f"got change {inp}"),inputs=input_txt)
        output_txt =gr.Textbox()

    btn = gr.Button("Run")
    btn.click(fn=predict,inputs=input_txt,outputs=output_txt)    

    flag= gr.Button("Flag")
    btn.click(fn=flag,inputs=[input_txt,output_txt])

demo.launch()    