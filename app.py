import gradio as gr
from self_rag import self_rag_pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import tempfile, os
from textwrap import wrap
# ==================== Interaction Function ====================

def interact(code_input: str, query: str, mode: str, role: str = ""):
    """Run the Self-RAG pipeline and prepare Markdown + Optional PDF."""
    if code_input.strip():
        print("📄  User provided a code snippet; adding to query context.")
        query += f" (Code snippet: {code_input[:200]}...)"

    result = self_rag_pipeline(query, mode=mode, role=role if mode == "recruiter" else "")

    # -------- Build step-by-step markdown --------
    steps_markdown = []
    for step in result["steps"]:
        name = step["step"].strip().lower()
        if any(keyword in name for keyword in ["critique", "refinement", "retrieve","retrieval"]):
            continue
        steps_markdown.append(f"### {step['step']}\n{step['content']}")

    final_answer = f"## 🧠 Final Answer\n{result['final_answer']}"
    full_markdown = "\n\n---\n\n".join(steps_markdown + [final_answer])

    # -------- Optional PDF for recruiter mode --------
    pdf_file = None
    if mode == "recruiter" and result.get("score") is not None:
        pdf_buffer = io.BytesIO()
        p = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        margin_x, margin_y = 100, 60
        y = height - margin_y

    # -------- Header --------
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin_x, y, "ReflectCode Review")
        y -= 20

        p.setFont("Helvetica", 12)
        p.drawString(margin_x, y, f"Query: {query[:80]}...")
        y -= 20
        p.drawString(margin_x, y, f"Score: {result['score']}/10")
        y -= 30

    # -------- Wrapped Content --------
        final_text = f"Final Notes:\n\n{result['final_answer']}"
        usable_width = width - 2 * margin_x       
        char_width = 6.0                          
        wrap_width = max(60, int(usable_width / char_width))
        wrapped_lines = []
        for line in final_text.splitlines():
            wrapped_lines.extend(wrap(line, wrap_width))
        p.setFont("Helvetica", 11)
        line_height = 14
        for line in wrapped_lines:
            if y < margin_y:
                p.showPage()
                y = height - margin_y
                p.setFont("Helvetica", 11)
            p.drawString(margin_x, y, line)
            y -= line_height
        p.save()
        pdf_buffer.seek(0)
        tmp_path = os.path.join(tempfile.gettempdir(), "ReflectCode_Review.pdf")
        with open(tmp_path, "wb") as f:
            f.write(pdf_buffer.getvalue())
        pdf_file = tmp_path
    return full_markdown, pdf_file


# ==================== Gradio Interface ====================

def build_interface():
    theme = gr.themes.Soft()

    with gr.Blocks(theme=theme, title="ReflectCode: Self-RAG Code Reviewer") as iface:
        gr.Markdown(
            """
            # 🔍 ReflectCode: AI Code Debugger & Interview Reviewer  
            Powered by **Self-RAG** — watch it retrieve, self‑critique, and refine answers live!
            """
        )

        with gr.Row():
            code_input = gr.Textbox(
                label="Paste Code Snippet (optional)",
                lines=5,
                placeholder="def buggy_func(x): return x*2 if x else None  # Edge case?"
            )
            query = gr.Textbox(
                label="Your Query",
                placeholder="E.g., 'Explain overfitting' or 'Review for ML Engineer'"
            )

        with gr.Row():
            mode = gr.Dropdown(
                choices=["dev", "recruiter"],
                label="Mode",
                value="dev"
            )
            role = gr.Textbox(
                label="Target Role (for Recruiter mode)",
                placeholder="E.g., 'Senior Dev'"
            )

        submit_btn = gr.Button("🚀 Run Reflection")

        # Outputs
        output_md = gr.Markdown(label="Reflection Steps")
        output_file = gr.File(label="📄 Download Review PDF (Recruiter mode only)", interactive=False)

        # Event
        submit_btn.click(
            fn=interact,
            inputs=[code_input, query, mode, role],
            outputs=[output_md, output_file],
            show_progress=True,
        )

        gr.Markdown(
            """
            ---
            **Built with LangChain + Groq**  
            Open‑source: [GitHub → Abhishek-622/reflectcode](https://github.com/Abhishek-622/reflectcode-self-rag)
            """
        )

    return iface


# ==================== Main Launcher ====================

if __name__ == "__main__":
    app = build_interface()
    app.launch(share=True)  