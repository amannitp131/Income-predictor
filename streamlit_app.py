import streamlit as st
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


st.set_page_config(page_title="Notebook Runner")
st.title("Run and View Notebook: App.ipynb")

nb_default = "App.ipynb"
nb_path = st.text_input("Notebook path", nb_default)

if st.button("Run notebook"):
    path = Path(nb_path)
    if not path.exists():
        st.error(f"Notebook not found: {path}")
    else:
        st.info("Executing notebook — this may take a while...")
        try:
            nb = nbformat.read(str(path), as_version=4)
        except Exception as e:
            st.error(f"Failed to read notebook: {e}")
        else:
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            try:
                ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
            except Exception as e:
                st.error(f"Execution failed: {e}")
            else:
                exporter = HTMLExporter()
                body, resources = exporter.from_notebook_node(nb)
                st.success("Execution finished. Rendering output below.")
                st.components.v1.html(body, height=800, scrolling=True)

st.write("\n---\n")
st.markdown("If you only want to view the notebook without running, open the `.ipynb` file directly.")
