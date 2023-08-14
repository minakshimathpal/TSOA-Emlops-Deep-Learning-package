import gradio as gr

def greet(name:str,last_name:str)-> str:
    return "Hello " + name +last_name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo = gr.Interface(fn=greet, inputs=[gr.Textbox(),gr.Textbox()], outputs="text")
    
demo.launch()
