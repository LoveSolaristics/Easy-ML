mkdir -p ~/.streamlit
echo "[server]
headless = true
port = $PORT
enableCORS = false

[runner]
magicEnabled = false
" > ~/.streamlit/config.toml
