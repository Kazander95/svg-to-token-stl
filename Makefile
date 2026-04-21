.PHONY: dev install sync clean

# Run the Streamlit UI with hot-reload.
dev:
	uv run streamlit run streamlit_app.py

# Install / sync dependencies.
install sync:
	uv sync

clean:
	rm -rf .venv __pycache__ *.egg-info
