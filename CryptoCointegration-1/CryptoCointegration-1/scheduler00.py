import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import logging

from pairs import analyze_cointegration_pairs  # Предполагаемая функция анализа
from database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def scheduled_job():
    logger.info(f"Запуск анализа коинтеграции: {datetime.now()}")
    try:
        await analyze_cointegration_pairs()
        logger.info("Анализ коинтеграции завершен успешно")
    except Exception as e:
        logger.error(f"Ошибка при анализе коинтеграции: {e}")

def start_scheduler():
    scheduler = AsyncIOScheduler()
    # Добавляем задачи на 02:00, 06:00, 10:00, 14:00, 18:00, 22:00
    scheduler.add_job(scheduled_job, 'cron', hour='2,6,10,14,18,22', minute=0)
    scheduler.start()
    logger.info("Планировщик запущен")

if __name__ == "__main__":
    start_scheduler()
    # Чтобы не завершать скрипт сразу
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
