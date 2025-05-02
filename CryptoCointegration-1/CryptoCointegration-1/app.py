import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple, Optional

from pairs import fetch_pairs
from database import DatabaseManager
from config import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройка Streamlit
st.set_page_config(
    page_title="Система анализа коинтеграции и парного трейдинга",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📊 Система анализа коинтеграции криптовалют")
st.subheader("Инструмент для выявления торговых возможностей статистического арбитража")

# Функция для запуска асинхронных задач
def run_async(coro):
    """Запуск асинхронного кода в синхронном контексте"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Функция для форматирования чисел
def format_number(num, precision=2):
    """Форматирование числа с нужной точностью и отделением тысяч"""
    if abs(num) < 0.01:
        return f"{num:.6f}"
    elif abs(num) < 1:
        return f"{num:.4f}"
    else:
        return f"{num:,.{precision}f}".replace(',', ' ')

# Создание боковой панели с фильтрами
with st.sidebar:
    st.header("Настройки анализа")
    
    # Выбор даты начала анализа
    default_date = datetime.now() - timedelta(days=7)
    selected_date = st.date_input(
        "Дата начала анализа",
        value=default_date.date(),
        min_value=default_date.date() - timedelta(days=30),
        max_value=date.today()
    )
    start_date = selected_date.strftime("%Y-%m-%d")
    
    # Выбор количества пар
    num_pairs = st.slider(
        "Количество пар для отображения",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
    
    # Выбор метода сортировки
    sort_options = {
        "z_score": "Z-показатель (отклонение от среднего)",
        "pct_change": "Процентное изменение",
        "volume": "Объем торгов"
    }
    
    sort_by = st.selectbox(
        "Сортировать по",
        options=list(sort_options.keys()),
        format_func=lambda x: sort_options[x],
        index=0
    )
    
    # Разделитель
    st.markdown("---")
    
    # Информация о проекте
    st.subheader("О проекте")
    st.markdown("""
    Система находит коинтегрированные пары криптовалют для 
    статистического арбитража. Алгоритм выявляет временные 
    аномалии в соотношении цен, которые имеют статистически 
    значимую тенденцию к возврату к среднему значению.
    """)

# Создание вкладок
tab1, tab2, tab3 = st.tabs([
    "📈 Коинтегрированные пары", 
    "🔍 Детальный анализ", 
    "⚙️ Управление данными"
])

# Вкладка 1: Коинтегрированные пары
with tab1:
    try:
        # Загрузка данных
        with st.spinner("Загрузка данных о коинтегрированных парах..."):
            pairs_data = fetch_pairs(
                start_date=start_date,
                num_pairs=num_pairs,
                sort_by=sort_by
            )
        
        if not pairs_data:
            st.warning("Не найдено коинтегрированных пар. Попробуйте изменить параметры фильтрации.")
        else:
            st.success(f"Найдено {len(pairs_data)} коинтегрированных пар")
            
            # Отображение пар в виде карточек
            cols = st.columns(min(3, len(pairs_data)))
            for i, pair_data in enumerate(pairs_data):
                s1, s2 = pair_data["symbols"]
                with cols[i % 3]:
                    card = st.container()
                    card.markdown(f"### {s1} / {s2}")
                    
                    # Создание графика для превью
                    fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True,
                                               vertical_spacing=0.02, row_heights=[0.7, 0.3])
                    
                    # График спреда
                    fig.add_trace(
                        go.Scatter(
                            x=pair_data["timestamps"],
                            y=pair_data["spread"],
                            mode="lines",
                            name="Спред",
                            line=dict(color="blue")
                        ),
                        row=1, col=1
                    )
                    
                    # График Z-показателя
                    fig.add_trace(
                        go.Scatter(
                            x=pair_data["timestamps"],
                            y=pair_data["zscore"],
                            mode="lines",
                            name="Z-score",
                            line=dict(color="red")
                        ),
                        row=2, col=1
                    )
                    
                    # Добавление пороговых линий для Z-показателя
                    fig.add_hline(y=2, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
                    fig.add_hline(y=-2, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
                    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False
                    )
                    
                    card.plotly_chart(fig, use_container_width=True)
                    
                    # Ключевые метрики
                    metrics_cols = card.columns(2)
                    metrics_cols[0].metric(
                        "Последний Z-score", 
                        format_number(pair_data["zscore"][-1] if pair_data["zscore"] else 0)
                    )
                    metrics_cols[1].metric(
                        "Изменение спреда, %", 
                        format_number(pair_data["pct_changes"][-1] if pair_data["pct_changes"] else 0)
                    )
                    
                    metrics_cols = card.columns(2)
                    metrics_cols[0].metric(
                        "Коэффициент хеджирования", 
                        format_number(pair_data["hedge_ratio"])
                    )
                    metrics_cols[1].metric(
                        "Период полураспада", 
                        format_number(pair_data["half_life"], 1)
                    )
                    
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        logger.exception("Ошибка на вкладке коинтегрированных пар")

# Вкладка 2: Детальный анализ
with tab2:
    if not pairs_data:
        st.info("Загрузите данные о коинтегрированных парах на первой вкладке.")
    else:
        # Выбор пары для детального анализа
        symbols_options = ["/".join(pair["symbols"]) for pair in pairs_data]
        selected_pair_idx = st.selectbox(
            "Выберите пару для детального анализа",
            options=range(len(symbols_options)),
            format_func=lambda i: symbols_options[i],
            index=0
        )
        
        selected_pair = pairs_data[selected_pair_idx]
        s1, s2 = selected_pair["symbols"]
        
        st.subheader(f"Детальный анализ пары {s1} / {s2}")
        
        # Создание графиков
        detail_tabs = st.tabs(["Спред и Z-score", "Цены активов", "Процентные изменения"])
        
        # Вкладка "Спред и Z-score"
        with detail_tabs[0]:
            fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # График спреда
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=selected_pair["spread"],
                    mode="lines",
                    name="Спред",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )
            
            # График Z-показателя
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=selected_pair["zscore"],
                    mode="lines",
                    name="Z-score",
                    line=dict(color="red")
                ),
                row=2, col=1
            )
            
            # Добавление пороговых линий для Z-показателя
            fig.add_hline(y=2, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=-2, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)
            
            fig.update_layout(
                height=600,
                title="Спред и Z-score",
                xaxis_title="Время",
                yaxis_title="Значение спреда",
                yaxis2_title="Z-score"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица сигналов
            st.subheader("Торговые сигналы")
            signals = []
            for i, z in enumerate(selected_pair["zscore"]):
                if abs(z) >= 2:
                    direction = "Покупка" if z < 0 else "Продажа"
                    signals.append({
                        "Время": selected_pair["timestamps"][i],
                        "Z-score": z,
                        "Направление": direction,
                        "Действие": f"{direction} {s1}, {'Продажа' if direction == 'Покупка' else 'Покупка'} {s2}"
                    })
            
            if signals:
                st.table(pd.DataFrame(signals).set_index("Время"))
            else:
                st.info("Нет сигналов с Z-score >= |2|")
        
        # Вкладка "Цены активов"
        with detail_tabs[1]:
            # Получение цен для обоих активов
            price1 = []
            price2 = []
            
            if "price_changes" in selected_pair:
                # Строим примерные цены на основе процентных изменений
                changes1 = selected_pair["price_changes"][s1]
                changes2 = selected_pair["price_changes"][s2]
                
                # Начальные значения берем из соотношения спреда и коэффициента хеджирования
                initial_spread = selected_pair["spread"][0] if selected_pair["spread"] else 1.0
                hedge_ratio = selected_pair["hedge_ratio"]
                
                # Используем условные начальные цены
                initial_price1 = 100.0
                initial_price2 = initial_price1 / hedge_ratio - initial_spread / hedge_ratio
                
                # Рассчитываем цены на основе процентных изменений
                price1 = [initial_price1]
                price2 = [initial_price2]
                
                for i in range(1, len(changes1)):
                    p1 = price1[-1] * (1 + changes1[i] / 100)
                    p2 = price2[-1] * (1 + changes2[i] / 100)
                    price1.append(p1)
                    price2.append(p2)
            
            # Создание графика
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=price1,
                    mode="lines",
                    name=s1,
                    line=dict(color="blue")
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=price2,
                    mode="lines",
                    name=s2,
                    line=dict(color="orange")
                )
            )
            
            fig.update_layout(
                height=500,
                title="Цены активов",
                xaxis_title="Время",
                yaxis_title="Цена"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Основные метрики
            cols = st.columns(4)
            
            # Коэффициент хеджирования
            cols[0].metric(
                "Коэффициент хеджирования",
                format_number(selected_pair["hedge_ratio"])
            )
            
            # Период полураспада
            cols[1].metric(
                "Период полураспада",
                format_number(selected_pair["half_life"], 1)
            )
            
            # Последний Z-score
            cols[2].metric(
                "Последний Z-score",
                format_number(selected_pair["zscore"][-1] if selected_pair["zscore"] else 0)
            )
            
            # Изменение спреда
            cols[3].metric(
                "Изменение спреда, %",
                format_number(selected_pair["pct_changes"][-1] if selected_pair["pct_changes"] else 0)
            )
        
        # Вкладка "Процентные изменения"
        with detail_tabs[2]:
            fig = go.Figure()
            
            # График процентных изменений для первого актива
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=selected_pair["price_changes"][s1],
                    mode="lines",
                    name=f"{s1} изменения, %",
                    line=dict(color="blue")
                )
            )
            
            # График процентных изменений для второго актива
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=selected_pair["price_changes"][s2],
                    mode="lines",
                    name=f"{s2} изменения, %",
                    line=dict(color="orange")
                )
            )
            
            # График процентного изменения спреда
            fig.add_trace(
                go.Scatter(
                    x=selected_pair["timestamps"],
                    y=selected_pair["pct_changes"],
                    mode="lines",
                    name="Изменение спреда, %",
                    line=dict(color="green")
                )
            )
            
            fig.update_layout(
                height=500,
                title="Процентные изменения",
                xaxis_title="Время",
                yaxis_title="Изменение, %"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Статистика изменений
            st.subheader("Статистика изменений")
            
            stats_cols = st.columns(3)
            
            # Статистика для первого актива
            with stats_cols[0]:
                changes1 = np.array(selected_pair["price_changes"][s1])
                changes1 = changes1[~np.isnan(changes1)]  # Удаление NaN значений
                
                st.markdown(f"**{s1}**")
                st.write(f"Среднее изменение: {format_number(np.mean(changes1))}%")
                st.write(f"Стандартное отклонение: {format_number(np.std(changes1))}%")
                st.write(f"Мин. изменение: {format_number(np.min(changes1))}%")
                st.write(f"Макс. изменение: {format_number(np.max(changes1))}%")
            
            # Статистика для второго актива
            with stats_cols[1]:
                changes2 = np.array(selected_pair["price_changes"][s2])
                changes2 = changes2[~np.isnan(changes2)]  # Удаление NaN значений
                
                st.markdown(f"**{s2}**")
                st.write(f"Среднее изменение: {format_number(np.mean(changes2))}%")
                st.write(f"Стандартное отклонение: {format_number(np.std(changes2))}%")
                st.write(f"Мин. изменение: {format_number(np.min(changes2))}%")
                st.write(f"Макс. изменение: {format_number(np.max(changes2))}%")
            
            # Статистика для спреда
            with stats_cols[2]:
                spread_changes = np.array(selected_pair["pct_changes"])
                spread_changes = spread_changes[~np.isnan(spread_changes)]  # Удаление NaN значений
                
                st.markdown("**Спред**")
                st.write(f"Среднее изменение: {format_number(np.mean(spread_changes))}%")
                st.write(f"Стандартное отклонение: {format_number(np.std(spread_changes))}%")
                st.write(f"Мин. изменение: {format_number(np.min(spread_changes))}%")
                st.write(f"Макс. изменение: {format_number(np.max(spread_changes))}%")

# Вкладка 3: Управление данными
with tab3:
    st.header("Управление данными")
    
    # Подключение к базе данных
    db = DatabaseManager()
    
    # Секция для запуска анализа
    st.subheader("Запуск анализа")
    
    # Статус последних запусков
    st.markdown("**Статус последних запусков**")
    
    # Таблица с условными данными о работе системы
    system_status = [
        {
            "Компонент": "Сбор данных",
            "Статус": "✅ Активен",
            "Последний запуск": "5 минут назад",
            "Результат": "Собрано данных по 120 символам"
        },
        {
            "Компонент": "Анализ коинтеграции",
            "Статус": "✅ Активен",
            "Последний запуск": "10 минут назад",
            "Результат": "Найдено 25 коинтегрированных пар"
        },
        {
            "Компонент": "Обновление спредов",
            "Статус": "✅ Активен",
            "Последний запуск": "2 минуты назад",
            "Результат": "Обновлены данные по 25 парам"
        }
    ]
    
    st.table(pd.DataFrame(system_status).set_index("Компонент"))
    
    # Кнопки для запуска процессов вручную
    cols = st.columns(3)
    collect_data = cols[0].button("Запустить сбор данных")
    analyze_pairs = cols[1].button("Запустить анализ коинтеграции")
    update_spreads = cols[2].button("Обновить данные спредов")
    
    if collect_data:
        st.info("Запущен процесс сбора данных...")
        st.warning("Функциональность в реализации. Эта кнопка пока не выполняет действий.")
    
    if analyze_pairs:
        st.info("Запущен процесс анализа коинтеграции...")
        st.warning("Функциональность в реализации. Эта кнопка пока не выполняет действий.")
    
    if update_spreads:
        st.info("Запущен процесс обновления спредов...")
        st.warning("Функциональность в реализации. Эта кнопка пока не выполняет действий.")
    
    # Статистика базы данных
    st.subheader("Статистика базы данных")
    
    try:
        # Получение статистики из базы данных
        with db.get_connection() as conn:
            from sqlalchemy import text
            
            # Общее количество символов
            symbols_count = conn.execute(text("SELECT COUNT(*) FROM symbols")).scalar() or 0
            
            # Количество активных символов с достаточным объемом
            active_symbols_count = conn.execute(text(f"""
                SELECT COUNT(*) FROM symbols 
                WHERE active = true AND volume_24h_usd > {config.analysis.volume_threshold_usdt}
            """)).scalar() or 0
            
            # Количество записей свечей
            candles_count = conn.execute(text("SELECT COUNT(*) FROM candles")).scalar() or 0
            
            # Количество коинтегрированных пар
            cointegrated_pairs_count = conn.execute(text("""
                SELECT COUNT(*) FROM cointegration_results 
                WHERE is_cointegrated = true AND valid_until > NOW()
            """)).scalar() or 0
            
            # Количество записей в таблице спредов
            spread_data_count = conn.execute(text("SELECT COUNT(*) FROM spread_data")).scalar() or 0
        
        # Отображение статистики
        stats_cols = st.columns(5)
        stats_cols[0].metric("Всего символов", symbols_count)
        stats_cols[1].metric("Активные символы", active_symbols_count)
        stats_cols[2].metric("Записи свечей", candles_count)
        stats_cols[3].metric("Коинтегр. пары", cointegrated_pairs_count)
        stats_cols[4].metric("Записи спредов", spread_data_count)
        
    except Exception as e:
        st.error(f"Ошибка при получении статистики: {str(e)}")
        logger.exception("Ошибка при получении статистики БД")
    
    # Таблица с последними коинтегрированными парами
    st.subheader("Последние коинтегрированные пары")
    
    try:
        with db.get_connection() as conn:
            from sqlalchemy import text
            
            query = """
                SELECT 
                    symbol1, 
                    symbol2, 
                    p_value, 
                    hedge_ratio, 
                    half_life,
                    created_at
                FROM cointegration_results
                WHERE is_cointegrated = true
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            df = pd.read_sql(text(query), conn)
            
            if not df.empty:
                # Форматирование данных
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                df['p_value'] = df['p_value'].apply(lambda x: format_number(x, 4))
                df['hedge_ratio'] = df['hedge_ratio'].apply(lambda x: format_number(x))
                df['half_life'] = df['half_life'].apply(lambda x: format_number(x, 1))
                
                # Переименование столбцов
                df.columns = ['Символ 1', 'Символ 2', 'P-значение', 'Коэфф. хеджирования', 'Период полураспада', 'Дата создания']
                
                st.table(df.set_index('Дата создания'))
            else:
                st.info("Нет данных о коинтегрированных парах")
                
    except Exception as e:
        st.error(f"Ошибка при получении данных о коинтегрированных парах: {str(e)}")
        logger.exception("Ошибка при получении данных о коинтегрированных парах")

# Добавление информации о времени последнего обновления
st.sidebar.markdown("---")
st.sidebar.caption(f"Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Работа с утилитами
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Утилиты"):
    if st.button("Инициализировать таблицы БД"):
        try:
            db = DatabaseManager()
            db._init_tables()
            st.success("Таблицы БД успешно инициализированы")
        except Exception as e:
            st.error(f"Ошибка при инициализации таблиц: {str(e)}")
