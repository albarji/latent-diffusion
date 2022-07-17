import gradio as gr

from gradioapp.txt2img import render_image

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""
            # Latent Diffusion Models
            Write a prompt to generate images illustrating that prompt.
            """
        )
        text_input = gr.Textbox()
        render_button = gr.Button("Render")
    output_images = []
    with gr.Row():
        out1 = gr.Image(shape=[256, 256])
        out2 = gr.Image(shape=[256, 256])
        out3 = gr.Image(shape=[256, 256])
        out4 = gr.Image(shape=[256, 256])
    with gr.Row():
        out5 = gr.Image(shape=[256, 256])
        out6 = gr.Image(shape=[256, 256])
        out7 = gr.Image(shape=[256, 256])
        out8 = gr.Image(shape=[256, 256])

    # for _ in range(3):
    #     with gr.Column():
    #         for _ in range(3):
    #             with gr.Row():
    #                 output_image = gr.Image(shape=[256, 256])
    #                 output_images.append(output_image)

    # render_button.click(render_image, inputs=text_input, outputs=output_images)
    render_button.click(
        render_image,
        inputs=text_input,
        outputs=[out1, out2, out3, out4, out5, out6, out7, out8]
    )

demo.launch(enable_queue=True, share=True)
