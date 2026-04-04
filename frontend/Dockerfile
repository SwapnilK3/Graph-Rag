# Lightweight frontend server using Python's http.server
FROM python:3.10-slim

WORKDIR /app

# Copy static files
COPY index.html .
COPY app.js .

# Expose port 3000
EXPOSE 3000

# Run a simple HTTP server
CMD ["python", "-m", "http.server", "3000"]
