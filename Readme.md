# AI Content Generation Service for Marketplaces

Дипломная работа: Сервис для генерации креативного контента с помощью ИИ для продавцов маркетплейсов.

## Описание

Комплексное решение для автоматизации создания визуального контента для товаров с использованием:
- **FLUX.1-dev** - диффузионная модель для генерации изображений с поддержкой LoRA адаптеров
- **Qwen2.5-VL-72B-Instruct** - визуально-языковая модель для описания изображений
- **Qwen2.5-32B-Instruct** - языковая модель для улучшения текстовых запросов
- **REST API** - FastAPI сервис для интеграции всех модулей
- **Telegram Bot** - пользовательский интерфейс

## Архитектура

### Основные модули:
1. **Модуль генерации изображений** (`src/models/flux_module.py`) - FLUX.1-dev с LoRA
2. **Модуль описания изображений** (`src/models/vlm_module.py`) - VLM для автоматического создания описаний
3. **Модуль улучшения запросов** (`src/models/llm_module.py`) - LLM для обработки пользовательских промптов
4. **Модуль обучения** (`src/models/training_module.py`) - LoRA обучение на FLUX.1-dev
5. **REST API** (`src/api/`) - FastAPI сервис с эндпоинтами
6. **Telegram Bot** (`src/telegram_bot/`) - интерфейс для пользователей

### База данных (PostgreSQL):
- `users` - пользователи
- `image_generations` - история генераций
- `model_trainings` - история обучений
- `lora_adapters` - LoRA адаптеры

### S3 хранилище:
```
/users/{user_id}/generated_images/    # сгенерированные изображения
/users/{user_id}/lora_adapters/       # веса LoRA адаптеров
/users/{user_id}/train_data/          # обучающие данные
```

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd diploma-2
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Настройка окружения
Создайте файл `.env` в корне проекта:
```env
# API настройки
API_HOST=0.0.0.0
API_PORT=8000

# База данных
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_content_db

# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=ai-content-storage

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token

# Модели
FLUX_MODEL_PATH=black-forest-labs/FLUX.1-dev
VLM_MODEL_PATH=Qwen/Qwen2-VL-72B-Instruct
LLM_MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
```

### 5. Инициализация базы данных
```bash
# Создайте базу данных PostgreSQL
createdb ai_content_db

# Таблицы создадутся автоматически при первом запуске
```

### 6. Запуск сервиса

#### REST API:
```bash
python -m src.main
```
API будет доступно по адресу: http://localhost:8000
Документация: http://localhost:8000/docs

#### Telegram Bot:
```bash
python -m src.telegram_bot.main
```

## API Эндпоинты

### Основные эндпоинты:

- `POST /api/v1/generate_image` - генерация изображений
- `POST /api/v1/generate_enhanced` - генерация с улучшением промпта
- `POST /api/v1/describe_image` - описание изображений (VLM)
- `POST /api/v1/improve_prompt` - улучшение промпта (LLM)
- `POST /api/v1/train` - обучение LoRA адаптера
- `GET /api/v1/health` - проверка здоровья сервиса
- `GET /api/v1/status` - статус всех компонентов

### Примеры запросов:

#### Генерация изображения:
```json
POST /api/v1/generate_image
{
  "prompt": "Стильная упаковка кофе на белом фоне",
  "num_images": 1,
  "lora_path": "path/to/lora/weights"
}
```

#### Обучение модели:
```json
POST /api/v1/train
{
  "train_data": [
    {
      "image": "base64_encoded_image",
      "description": "Описание изображения"
    }
  ]
}
```

## Telegram Bot

Бот предоставляет интуитивный интерфейс для:
- Генерации изображений (стандартная и улучшенная)
- Обучения собственных моделей
- Просмотра истории операций

### Команды:
- `/start` - начало работы
- `/help` - справка

## Технические требования

### Минимальные требования:
- Python 3.11+
- PostgreSQL 12+
- 16GB RAM (для загрузки моделей)
- NVIDIA GPU с 24GB+ VRAM (рекомендуется)

### Рекомендуемые требования:
- NVIDIA A100/H100 GPU
- 64GB+ RAM
- SSD хранилище

## Разработка

### Структура проекта:
```
src/
├── api/              # REST API
├── config/           # Конфигурация
├── database/         # Модели БД
├── models/           # AI модули
├── storage/          # S3 клиент
├── telegram_bot/     # Telegram бот
└── main.py          # Точка входа
```

### Тестирование:
```bash
pytest tests/
```

### Форматирование кода:
```bash
black src/
flake8 src/
```

## Лицензия

Дипломная работа - все права защищены.

## Контакты

Автор: [Ваше имя]
Email: [ваш email]
