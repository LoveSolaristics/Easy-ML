mkdir -p ~/.streamlit
echo "[server]
headless = true
port = $PORT
enableCORS = false

[runner]
magicEnabled = false

[logger]

level = 'error'
" > ~/.streamlit/config.toml
