FROM python:3.14
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY temp.py .
EXPOSE 8000
CMD ["uvicorn","temp:app","--host","0.0.0.0","--port","8000"]

