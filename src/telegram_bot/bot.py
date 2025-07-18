"""
Карточки товаров AI
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

class BotStates(StatesGroup):
    MAIN_MENU = State()
    GENERATING_IMAGE = State()
    TRAINING_MODEL = State()
    UPLOADING_TRAINING_DATA = State()
    WAITING_PRODUCT_NAME = State()
    WAITING_MODEL_SELECTION = State()
    WAITING_GENERATION_PROMPT = State()

class TelegramBot:
    """Карточки товаров AI"""
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
        self._setup_handlers()
        logger.info("Telegram bot initialized")

    def _setup_handlers(self):
        """Setup message handlers"""
        self.dp.message.register(self.start_command, Command("start"))
        self.dp.message.register(self.help_command, Command("help"))
        
        self.dp.message.register(
            self.generate_image_handler,
            F.text == "🧠 Сгенерировать товар"
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
        
        self.dp.message.register(
            self.process_product_name,
            StateFilter(BotStates.WAITING_PRODUCT_NAME)
        )
        self.dp.message.register(
            self.process_training_data,
            StateFilter(BotStates.UPLOADING_TRAINING_DATA),
            F.content_type.in_(['photo', 'text'])
        )
        self.dp.message.register(
            self.process_model_selection,
            StateFilter(BotStates.WAITING_MODEL_SELECTION)
        )
        self.dp.message.register(
            self.process_generation_prompt,
            StateFilter(BotStates.WAITING_GENERATION_PROMPT)
        )
        
        self.dp.callback_query.register(
            self.generate_again_callback,
            F.data == "generate_again"
        )
        self.dp.callback_query.register(
            self.main_menu_callback,
            F.data == "main_menu"
        )

    def get_main_keyboard(self) -> ReplyKeyboardMarkup:
        """Get main menu keyboard"""
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="🧠 Обучить модель")],
                [KeyboardButton(text="🧠 Сгенерировать товар")],
                [KeyboardButton(text="📝 История"), KeyboardButton(text="❓ Помощь")]
            ],
            resize_keyboard=True,
            persistent=True
        )
        return keyboard

    def get_generation_result_keyboard(self) -> InlineKeyboardMarkup:
        """Get generation result keyboard"""
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Сгенерировать ещё раз", callback_data="generate_again")],
                [InlineKeyboardButton(text="🏠 В начальное меню", callback_data="main_menu")]
            ]
        )
        return keyboard

    async def start_command(self, message: types.Message, state: FSMContext):
        """Handle /start command"""
        await state.set_state(BotStates.MAIN_MENU)
        welcome_text = """🧠 Добро пожаловать в демо бот сервиса генерации товаров!

Доступные функции:
🧠 Обучить модель - обучение на ваших данных
🧠 Сгенерировать товар - создание новых изображений

Выберите действие:"""
        await message.answer(
            welcome_text,
            reply_markup=self.get_main_keyboard()
        )

    async def help_command(self, message: types.Message):
        """Handle /help command"""
        await self.help_handler(message)

    async def help_handler(self, message: types.Message):
        """Handle help button"""
        help_text = """❓ Справка по использованию:

🧠 **Обучение модели:**
1. Нажмите "Обучить модель"
2. Введите название товара
3. Загружайте изображения товаров
4. Дождитесь завершения обучения

🧠 **Генерация товара:**
1. Нажмите "Сгенерировать товар"
2. Выберите обученную модель
3. Введите запрос на генерацию
4. Получите сгенерированное изображение

💡 **Советы:**
• Описывайте товары детально
• Для лучших результатов используйте 5-20 изображений для обучения"""
        await message.answer(help_text)

    async def train_model_handler(self, message: types.Message, state: FSMContext):
        """Handle model training button"""
        await state.set_state(BotStates.WAITING_PRODUCT_NAME)
        await message.answer("🧠 **Обучение модели**\n\n📝 Пожалуйста, напишите название товара для обучения:")

    async def process_product_name(self, message: types.Message, state: FSMContext):
        """Process product name for training"""
        if not message.text:
            await message.answer("❌ Пожалуйста, отправьте текстовое название товара.")
            return
        
        product_name = message.text
        await state.update_data(product_name=product_name, training_data=[], image_count=0)
        await state.set_state(BotStates.UPLOADING_TRAINING_DATA)
        
        await message.answer(
            f"✅ **Название товара: {product_name}**\n\n"
            f"📸 Теперь отправьте изображения для обучения.\n"
            f"Вы можете отправить несколько изображений одно за другим.\n"
            f"Когда закончите, напишите 'готово' или 'завершить'."
        )

    async def process_training_data(self, message: types.Message, state: FSMContext):
        """Process training data upload"""
        data = await state.get_data()
        training_data = data.get("training_data", [])
        product_name = data.get("product_name", "")
        image_count = data.get("image_count", 0)

        if message.photo:
            photo = message.photo[-1]
            file = await self.bot.get_file(photo.file_id)
            photo_data = await self.bot.download_file(file.file_path)
            base64_image = base64.b64encode(photo_data.read()).decode('utf-8')
            
            training_data.append({
                "image": base64_image,
                "description": product_name
            })
            image_count += 1
            
            await state.update_data(training_data=training_data, image_count=image_count)
            await message.answer(
                f"✅ **Изображение {image_count} получено!**\n"
                f"Продолжайте загружать изображения или напишите 'готово'."
            )

        elif message.text and message.text.lower() in ['готово', 'завершить']:
            if len(training_data) < 1:
                await message.answer("❌ Загрузите хотя бы одно изображение.")
                return
            await self.start_training(message, training_data, product_name, state)

    async def start_training(self, message: types.Message, training_data: list, product_name: str, state: FSMContext):
        """Start model training"""
        await message.answer(
            f"🧠 **Обучение модели началось**\n\n"
            f"📦 **Товар:** {product_name}\n"
            f"📸 **Изображений:** {len(training_data)}\n"
            f"⏰ **Ожидаемое время:** 30 минут"
        )
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/api/v1/train",
                    json={"train_data": training_data}
                )
            
            if response.status_code == 200:
                result = response.json()
                await message.answer(
                    f"🎉 **Модель обучилась успешно!**\n\n"
                    f"📦 **Товар:** {product_name}\n"
                    f"✅ **Модель готова к использованию**"
                )
            else:
                error_msg = response.json().get("detail", "Неизвестная ошибка")
                await message.answer(f"❌ Ошибка обучения: {error_msg}")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            await message.answer("❌ Произошла ошибка при обучении модели.")
        
        await state.set_state(BotStates.MAIN_MENU)

    async def generate_image_handler(self, message: types.Message, state: FSMContext):
        """Handle image generation button"""
        await state.set_state(BotStates.WAITING_MODEL_SELECTION)
        await message.answer(
            "🧠 **Генерация товара**\n\n"
            "📝 Пожалуйста, напишите название модели для генерации:"
        )

    async def process_model_selection(self, message: types.Message, state: FSMContext):
        """Process model selection for generation"""
        if not message.text:
            await message.answer("❌ Пожалуйста, отправьте название модели.")
            return
        
        model_name = message.text
        await state.update_data(model_name=model_name)
        await state.set_state(BotStates.WAITING_GENERATION_PROMPT)
        
        await message.answer(
            f"✅ **Модель: {model_name}**\n\n"
            f"📝 Теперь напишите запрос на генерацию:"
        )

    async def process_generation_prompt(self, message: types.Message, state: FSMContext):
        """Process generation prompt"""
        if not message.text:
            await message.answer("❌ Пожалуйста, отправьте текстовый запрос.")
            return
        
        data = await state.get_data()
        model_name = data.get("model_name", "")
        prompt = message.text
        
        await message.answer("🎨 Генерирую изображение... Пожалуйста, подождите.")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/api/v1/generate_image",
                    json={
                        "prompt": prompt,
                        "num_images": 1
                    }
                )
            
            if response.status_code == 200:
                result = response.json()
                images = result.get("images", [])
                if images:
                    for i, base64_image in enumerate(images):
                        image_data = base64.b64decode(base64_image)
                        await message.answer_photo(
                            types.BufferedInputFile(image_data, filename=f"generated_{i+1}.png"),
                            caption=(
                                f"🎨 **Изображение сгенерировано!**\n\n"
                                f"📦 **Модель:** {model_name}\n"
                                f"📝 **Запрос:** {prompt}"
                            ),
                            reply_markup=self.get_generation_result_keyboard()
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

    async def generate_again_callback(self, callback: types.CallbackQuery, state: FSMContext):
        """Handle generate again callback"""
        data = await state.get_data()
        model_name = data.get("model_name", "")
        
        await state.set_state(BotStates.WAITING_GENERATION_PROMPT)
        await callback.message.answer(
            f"🔄 **Сгенерировать ещё раз**\n\n"
            f"📦 **Модель:** {model_name}\n"
            f"📝 Напишите новый запрос на генерацию:"
        )
        await callback.answer()

    async def main_menu_callback(self, callback: types.CallbackQuery, state: FSMContext):
        """Handle main menu callback"""
        await state.set_state(BotStates.MAIN_MENU)
        await callback.message.answer(
            "🏠 **В начальное меню**",
            reply_markup=self.get_main_keyboard()
        )
        await callback.answer()

    async def history_handler(self, message: types.Message):
        """Handle history button"""
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
