"""
Main entry point for Telegram Bot
"""
import asyncio
import logging
from src.telegram_bot.bot import create_bot
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("telegram_bot.log")
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to start the Telegram bot"""
    try:
        logger.info("Starting Telegram Bot...")
        
        # Create bot instance
        bot = create_bot()
        
        # Start polling
        await bot.start_polling()
        
    except Exception as e:
        logger.error(f"Error starting Telegram bot: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 