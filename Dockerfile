FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# Hugging Face Spaces 默认通信端口为 7860
EXPOSE 7860
ENV HOST=0.0.0.0
ENV PORT=7860

CMD ["python", "src/web.py", "--host", "0.0.0.0", "--port", "7860"]
