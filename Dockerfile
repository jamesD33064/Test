# 使用 PHP 8.2 作為基底映像檔
FROM php:8.2-fpm

# 安裝必要的系統套件
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libpng-dev \
    libonig-dev \
    libxml2-dev \
    zip \
    unzip \
    libzip-dev \
    nginx \
    && docker-php-ext-install pdo_mysql mbstring exif pcntl bcmath gd

# 安裝 Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# 設定工作目錄
WORKDIR /var/www/html

# 複製 Laravel 專案檔案
COPY ./AsiaYoTest .

# 安裝 Laravel 相依套件
ENV COMPOSER_ALLOW_SUPERUSER=1
RUN composer install --no-dev --no-interaction --optimize-autoloader

# 產生 Laravel 應用程式金鑰
RUN php artisan key:generate

# 建立 MySQL 連線設定
ENV DB_CONNECTION=mysql
ENV DB_HOST=mysql
ENV DB_PORT=3306
ENV DB_DATABASE=laravel
ENV DB_USERNAME=root
ENV DB_PASSWORD=password

CMD ["php", "artisan", "serve", "--host=0.0.0.0", "--port=8000"]