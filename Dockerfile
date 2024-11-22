# Gunakan image Python sebagai base image
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Menetapkan environment variable untuk Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Install Gunicorn
RUN pip install gunicorn

# Perintah untuk menjalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]