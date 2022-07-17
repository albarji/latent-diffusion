import gradio as gr

from gradioapp.txt2img import render_image

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""
            # Latent Diffusion Models
            Write a prompt to generate images illustrating that prompt.
            """
        )
        with gr.Column():
            text_input = gr.Textbox()
            steps = gr.Slider(minimum=50, maximum=500, value=250, step=10, label="Number of diffusion model steps")
            guidance = gr.Slider(minimum=0.0, maximum=15.0, value=5.0, step=1.0, label="Classifier-free guidance strength")
        render_button = gr.Button("Render")
    output_images = []
    for _ in range(2):
        with gr.Row():
            for _ in range(4):
                out = gr.Image(shape=[256, 256])
                output_images.append(out)

    render_button.click(
        render_image,
        inputs=[text_input, steps, guidance],
        outputs=output_images
    )

demo.launch(enable_queue=True, share=True)
