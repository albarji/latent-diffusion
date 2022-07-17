import gradio as gr

from gradioapp.txt2img import render_image

demo = gr.Interface(fn=render_image, inputs="text", outputs=["image"]*9)

demo.launch(enable_queue=True, share=True)
