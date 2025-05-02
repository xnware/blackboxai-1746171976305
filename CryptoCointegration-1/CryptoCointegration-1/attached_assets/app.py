import logging
from flask import Flask, render_template, request
from datetime import datetime, timezone
from config import Config
from utils import validate_date
from pairs import fetch_pairs

# Настройка логирования
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.debug = True  # включить режим отладки

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Unhandled exception:")
    return "500 — внутренняя ошибка сервера", 500

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Получаем дату из формы или GET-параметра
        if request.method == 'POST':
            user_date = request.form.get('start_date')
        else:
            user_date = request.args.get('start_date')

        if user_date and validate_date(user_date):
            date = user_date
        else:
            date = Config.DEFAULT_START_DATE or datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # Количество пар
        num_pairs = request.form.get('num_pairs') if request.method == 'POST' else request.args.get('num_pairs')
        num_pairs = int(num_pairs) if num_pairs else Config.DEFAULT_NUM_PAIRS
        try:
            num_pairs = int(num_pairs)
        except (ValueError, TypeError):
            num_pairs = Config.DEFAULT_NUM_PAIRS

        # Сортировка
        sort_by = (request.form.get('sort_by') if request.method == 'POST'
                   else request.args.get('sort_by', Config.DEFAULT_SORT_BY))
        
        # Количество периодов
        periods = (request.form.get('periods') if request.method == 'POST'
                   else request.args.get('periods', '100'))
        try:
            periods = int(periods)
        except (ValueError, TypeError):
            periods = 100

        logger.debug(f"Parameters - date: {date}, num_pairs: {num_pairs}, sort_by: {sort_by}, periods: {periods}")

        pairs = fetch_pairs(date, num_pairs, sort_by, periods)
        return render_template('index.html', pairs=pairs, date=date,
                               num_pairs=num_pairs, sort_by=sort_by, periods=periods)
    except Exception as e:
        logger.exception("Error in index route:")
        return f"Ошибка сервера: {e}", 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # для тестового HTTPS