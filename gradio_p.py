import gradio as gr
"""
Running on local URL:  http://127.0.0.1:7861
2023/12/17 13:55:01 [W] [service.go:132] login to server failed: dial tcp 44.237.78.176:7000: i/o timeout

Could not create share link. Please check your internet connection or our status page: https://status.gradio.app.
"""
def echo(text):
    return text

demo = gr.Interface(fn=echo, inputs="text", outputs="text")

demo.launch(share=True)