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
            steps = gr.Slider(minimum=50, maximum=500, value=250, step=10, label="Number of diffusion model steps: more steps means larger generation times, but better quality.")
            guidance = gr.Slider(minimum=0.0, maximum=15.0, value=5.0, step=1.0, label="Classifier-free guidance strength: at 1 the model generates images at random, larger values focus more on the prompt provided")
            blank_cond = gr.Number(value=0.0, label="Blank-conditioning strength: higher values produce darker images (just for testing purposes)")
        render_button = gr.Button("Render")
    output_images = []
    for _ in range(2):
        with gr.Row():
            for _ in range(4):
                out = gr.Image(shape=[256, 256])
                output_images.append(out)

    render_button.click(
        render_image,
        inputs=[text_input, steps, guidance, blank_cond],
        outputs=output_images,
    )

demo.launch(enable_queue=True, server_name="0.0.0.0")
