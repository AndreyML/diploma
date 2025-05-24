"""
Telegram Bot for AI Content Generation Service
Implementation based on the diploma work specification
"""
import logging
import base64
import io
from typing import Optional
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import httpx
from src.config.settings import settings
logger = logging.getLogger(__name__)
# Bot states
class BotStates(StatesGroup):
    MAIN_MENU = State()
    GENERATING_IMAGE = State()
    TRAINING_MODEL = State()
    UPLOADING_TRAINING_DATA = State()
class TelegramBot:
    """Telegram bot for AI content generation service"""
    def __init__(self, token: str, api_base_url: str):
        """
        Initialize Telegram bot
        Args:
            token: Telegram bot token
            api_base_url: Base URL of the REST API service
        """
        self.bot = Bot(token=token)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.api_base_url = api_base_url.rstrip('/')
        # Setup handlers
        self._setup_handlers()
        logger.info("Telegram bot initialized")
    def _setup_handlers(self):
        """Setup message handlers"""
        # Command handlers
        self.dp.message.register(self.start_command, Command("start"))
        self.dp.message.register(self.help_command, Command("help"))
        # Menu handlers
        self.dp.message.register(
            self.generate_image_handler,
            F.text == "🎨 Генерировать изображение"
        )
        self.dp.message.register(
            self.train_model_handler,
            F.text == "🧠 Обучить модель"
        )
        self.dp.message.register(
            self.history_handler,
            F.text == "📝 История"
        )
        self.dp.message.register(
            self.help_handler,
            F.text == "❓ Помощь"
        )
        # State handlers
        self.dp.message.register(
            self.process_image_prompt,
            StateFilter(BotStates.GENERATING_IMAGE)
        )
        self.dp.message.register(
            self.process_training_data,
            StateFilter(BotStates.UPLOADING_TRAINING_DATA),
            F.content_type.in_(['photo', 'text'])
        )
        # Callback handlers
        self.dp.callback_query.register(
            self.enhanced_generation_callback,
            F.data == "enhanced_generation"
        )
        self.dp.callback_query.register(
            self.standard_generation_callback,
            F.data == "standard_generation"
        )
    def get_main_keyboard(self) -> ReplyKeyboardMarkup:
        """Get main menu keyboard"""
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="🎨 Генерировать изображение")],
                [KeyboardButton(text="🧠 Обучить модель")],
                [KeyboardButton(text="📝 История"), KeyboardButton(text="❓ Помощь")]
            ],
            resize_keyboard=True,
            persistent=True
        )
        return keyboard
    def get_generation_keyboard(self) -> InlineKeyboardMarkup:
        """Get generation type selection keyboard"""
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="✨ Улучшенная генерация", callback_data="enhanced_generation")],
                [InlineKeyboardButton(text="🎯 Стандартная генерация", callback_data="standard_generation")]
            ]
        )
        return keyboard
    async def start_command(self, message: types.Message, state: FSMContext):
        """Handle /start command"""
        await state.set_state(BotStates.MAIN_MENU)
        welcome_text = """
🤖 Добро пожаловать в AI Content Generator!
Этот бот поможет вам создавать креативный контент для маркетплейсов с помощью искусственного интеллекта.
🎨 Функции:
• Генерация изображений товаров по текстовому описанию
• Обучение собственных моделей на ваших данных
• Улучшение текстовых запросов с помощью ИИ
Выберите действие в меню ниже:
        """
        await message.answer(
            welcome_text,
            reply_markup=self.get_main_keyboard()
        )
    async def help_command(self, message: types.Message):
        """Handle /help command"""
        await self.help_handler(message)
    async def help_handler(self, message: types.Message):
        """Handle help button"""
        help_text = """
❓ Справка по использованию:
🎨 **Генерация изображений:**
1. Нажмите "Генерировать изображение"
2. Выберите тип генерации:
   • Улучшенная - с автоматическим улучшением промпта
   • Стандартная - использует ваш промпт как есть
3. Отправьте текстовое описание желаемого изображения
4. Получите сгенерированные изображения
🧠 **Обучение модели:**
1. Нажмите "Обучить модель"
2. Загружайте изображения товаров с описаниями
3. Дождитесь завершения обучения
4. Используйте обученную модель для генерации
📝 **История:**
Просмотр ваших предыдущих генераций и обучений
💡 **Советы:**
• Описывайте товары детально (цвет, форма, стиль)
• Указывайте желаемый фон и освещение
• Для лучших результатов используйте 5-20 изображений для обучения
        """
        await message.answer(help_text)
    async def generate_image_handler(self, message: types.Message, state: FSMContext):
        """Handle image generation button"""
        await message.answer(
            "🎨 Выберите тип генерации:",
            reply_markup=self.get_generation_keyboard()
        )
    async def enhanced_generation_callback(self, callback: types.CallbackQuery, state: FSMContext):
        """Handle enhanced generation callback"""
        await state.set_state(BotStates.GENERATING_IMAGE)
        await state.update_data(generation_type="enhanced")
        await callback.message.answer(
            "✨ **Улучшенная генерация**\n\n"
            "Ваш промпт будет автоматически улучшен с помощью ИИ для получения лучших результатов.\n\n"
            "📝 Отправьте описание желаемого изображения товара:"
        )
        await callback.answer()
    async def standard_generation_callback(self, callback: types.CallbackQuery, state: FSMContext):
        """Handle standard generation callback"""
        await state.set_state(BotStates.GENERATING_IMAGE)
        await state.update_data(generation_type="standard")
        await callback.message.answer(
            "🎯 **Стандартная генерация**\n\n"
            "Изображение будет сгенерировано точно по вашему описанию.\n\n"
            "📝 Отправьте описание желаемого изображения товара:"
        )
        await callback.answer()
    async def process_image_prompt(self, message: types.Message, state: FSMContext):
        """Process image generation prompt"""
        if not message.text:
            await message.answer("❌ Пожалуйста, отправьте текстовое описание.")
            return
        data = await state.get_data()
        generation_type = data.get("generation_type", "standard")
        await message.answer("🎨 Генерирую изображение... Пожалуйста, подождите.")
        try:
            # Choose endpoint based on generation type
            endpoint = "/api/v1/generate_enhanced" if generation_type == "enhanced" else "/api/v1/generate_image"
            # Make API request
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base_url}{endpoint}",
                    json={
                        "prompt": message.text,
                        "num_images": 1
                    }
                )
            if response.status_code == 200:
                result = response.json()
                images = result.get("images", [])
                if images:
                    for i, base64_image in enumerate(images):
                        # Convert base64 to bytes
                        image_data = base64.b64decode(base64_image)
                        # Send image
                        await message.answer_photo(
                            types.BufferedInputFile(image_data, filename=f"generated_{i+1}.png"),
                            caption=f"🎨 Сгенерированное изображение\n📝 Промпт: {message.text[:100]}..."
                        )
                else:
                    await message.answer("❌ Не удалось сгенерировать изображение.")
            else:
                error_msg = response.json().get("detail", "Неизвестная ошибка")
                await message.answer(f"❌ Ошибка генерации: {error_msg}")
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            await message.answer("❌ Произошла ошибка при генерации изображения.")
        await state.set_state(BotStates.MAIN_MENU)
    async def train_model_handler(self, message: types.Message, state: FSMContext):
        """Handle model training button"""
        await state.set_state(BotStates.UPLOADING_TRAINING_DATA)
        await state.update_data(training_data=[])
        await message.answer(
            "🧠 **Обучение модели**\n\n"
            "Загружайте изображения товаров с описаниями для обучения.\n\n"
            "📝 Для каждого изображения:\n"
            "1. Отправьте фото товара\n"
            "2. Сразу после фото отправьте текстовое описание\n\n"
            "Когда загрузите все данные, отправьте команду /train для начала обучения.\n\n"
            "💡 Рекомендуется 5-20 изображений для качественного обучения."
        )
    async def process_training_data(self, message: types.Message, state: FSMContext):
        """Process training data upload"""
        data = await state.get_data()
        training_data = data.get("training_data", [])
        if message.photo:
            # Get the largest photo
            photo = message.photo[-1]
            # Download photo
            file = await self.bot.get_file(photo.file_id)
            photo_data = await self.bot.download_file(file.file_path)
            # Convert to base64
            base64_image = base64.b64encode(photo_data.read()).decode('utf-8')
            # Store photo temporarily
            await state.update_data(pending_image=base64_image)
            await message.answer(
                "📸 Изображение получено! Теперь отправьте текстовое описание для этого изображения:"
            )
        elif message.text:
            if message.text.startswith("/train"):
                # Start training
                if len(training_data) < 1:
                    await message.answer("❌ Загрузите хотя бы одно изображение с описанием.")
                    return
                await self.start_training(message, training_data, state)
                return
            # Process description
            pending_image = data.get("pending_image")
            if pending_image:
                # Add training sample
                training_data.append({
                    "image": pending_image,
                    "description": message.text
                })
                await state.update_data(
                    training_data=training_data,
                    pending_image=None
                )
                await message.answer(
                    f"✅ Пара изображение-описание добавлена! "
                    f"Всего образцов: {len(training_data)}\n\n"
                    f"Продолжайте загружать данные или отправьте /train для начала обучения."
                )
            else:
                await message.answer(
                    "❌ Сначала отправьте изображение, затем описание.\n"
                    "Или отправьте /train для начала обучения."
                )
    async def start_training(self, message: types.Message, training_data: list, state: FSMContext):
        """Start model training"""
        await message.answer(
            f"🧠 Начинаю обучение модели на {len(training_data)} образцах...\n"
            f"⏳ Это может занять несколько минут."
        )
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/api/v1/train",
                    json={"train_data": training_data}
                )
            if response.status_code == 200:
                result = response.json()
                lora_url = result.get("lora_params_url", "")
                await message.answer(
                    "✅ **Обучение завершено успешно!**\n\n"
                    f"🔗 Ссылка на веса модели:\n`{lora_url}`\n\n"
                    "Теперь вы можете использовать обученную модель для генерации изображений.",
                    parse_mode="Markdown"
                )
            else:
                error_msg = response.json().get("detail", "Неизвестная ошибка")
                await message.answer(f"❌ Ошибка обучения: {error_msg}")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            await message.answer("❌ Произошла ошибка при обучении модели.")
        await state.set_state(BotStates.MAIN_MENU)
    async def history_handler(self, message: types.Message):
        """Handle history button"""
        # In a real implementation, this would fetch user's history from database
        await message.answer(
            "📝 **История операций**\n\n"
            "Здесь будет отображаться история ваших генераций и обучений.\n\n"
            "🔧 Функция в разработке..."
        )
    async def start_polling(self):
        """Start bot polling"""
        try:
            logger.info("Starting Telegram bot polling...")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Error in bot polling: {e}")
            raise
    async def stop(self):
        """Stop the bot"""
        await self.bot.session.close()
# Factory function to create bot instance
def create_bot(token: str = None, api_base_url: str = None) -> TelegramBot:
    """
    Create Telegram bot instance
    Args:
        token: Telegram bot token (from settings if None)
        api_base_url: API base URL (from settings if None)
    Returns:
        TelegramBot instance
    """
    if not token:
        token = settings.TELEGRAM_BOT_TOKEN
    if not api_base_url:
        api_base_url = f"http://{settings.API_HOST}:{settings.API_PORT}"
    if not token:
        raise ValueError("Telegram bot token is required")
    return TelegramBot(token, api_base_url)
