# AsiaYo Test

## 建置環境

```
cd AsiaYoTest
docker run -v $(pwd):/app -w /app composer composer install
docker-compose up -d
```