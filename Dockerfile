# 1. Use a stable, slim version of Python
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy only requirements first (Optimizes Docker Caching)
COPY requirements.txt .

# 4. Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project code
COPY . .

# 6. Open the port your app will run on
EXPOSE 5000

# 7. Start the app
CMD ["python", "app.py"]