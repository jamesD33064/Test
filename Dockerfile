# 使用 PHP 8.2 作為基底映像檔
FROM php:8.2-fpm

# 安裝必要的系統套件
RUN apt-get update && apt-get install -y \
    libonig-dev \
    libxml2-dev \
    libzip-dev \
    libpng-dev \
    git \
    unzip \
    && docker-php-ext-install zip mbstring exif pcntl bcmath gd

# 安裝 Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# 設定工作目錄
WORKDIR /var/www/html

# 複製 Laravel 專案檔案
COPY ./AsiaYoTest .
COPY ./AsiaYoTest/.env.example .env

# 產生 Laravel 應用程式金鑰
RUN php artisan key:generate

CMD ["php", "artisan", "serve", "--host=0.0.0.0", "--port=8000"]