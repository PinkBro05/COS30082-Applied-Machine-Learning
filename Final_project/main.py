"""
Main application for face recognition check-in system.
Uses the modular components from the face_modules package.
"""

import gradio as gr
from face_modules.check_in_system import process_check_in, process_registration

def main():
    """
    Main function to run the face recognition check-in application.
    """
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(sources=["webcam"], streaming=True)
                with gr.Row():
                    with gr.Column():
                        check_in_button = gr.Button("Check-in", variant="primary")
                    with gr.Column():
                        register_button = gr.Button("Register", variant="primary")
                        name_text = gr.Textbox(label="Name")
            with gr.Column(scale=1):
                result_annotated_img = gr.AnnotatedImage()
                eye_result = gr.Textbox(label="Eyes result")
                img_result = gr.Textbox(label="Image result")
                chk_result = gr.Textbox(label="Checked-in result")

        state = gr.State(value={
            "last_face": "",
            "taken_actions": set()
        })

        check_in_button.click(
            process_check_in,
            inputs=[image, state],
            outputs=[eye_result, img_result, chk_result, result_annotated_img, state]
        )

        register_button.click(
            process_registration,
            inputs=[image, name_text],
            outputs=[img_result, result_annotated_img]
        )

    demo.launch()

if __name__ == "__main__":
    main()
