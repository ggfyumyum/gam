# Always run Streamlit with the project venv (avoids accidentally using Anaconda)
& "$PSScriptRoot\.venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\horse.py"
