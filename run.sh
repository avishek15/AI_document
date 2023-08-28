docker run -p 80:8501 \
  -v ./conversation_cache:/app/conversation_cache \
  -v ./docs_cache:/app/docs_cache \
  -v ./memory_cache:/app/memory_cache \
  -v ./model_cache:/app/model_cache \
  -v ./upload_dir:/app/upload_dir \
  --rm -it ai_document:v0.1 /bin/bash