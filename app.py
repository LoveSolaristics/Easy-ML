import streamlit as st
from src.home import home_app
from src.data import data_app
from src.graphics import graphics_app
from src.model import model_app

from src.load_dataset import load_dataset

from multipage import save, MultiPage, start_app, clear_cache

st.set_page_config(
    page_title="Easy ML", page_icon='💙',
)

start_app()  # Clears the cache when the app is started
app = MultiPage()

# app.start_button = "Поехали!"
app.navbar_name = "Навигация"
app.next_page_button = "Следующая страница"
app.previous_page_button = "Предыдущая страница"

# Add all your application here

app.add_app("README.md", home_app)
app.add_app("Просмотр датасета через Pandas", data_app)
app.add_app("Построение графиков", graphics_app)
app.add_app("Обучение моделей", model_app)

# The main app
app.run()
