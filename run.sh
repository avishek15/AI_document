docker run -p 80:8501 \
  -v ./conversation_cache:conversation_cache \
  -v ./docs_cache:docs_cache \
  -v ./memory_cache:memory_cache \
  -v ./model_cache:model_cache \
  -v ./upload_dir:upload_dir \
  --rm -it AI_document:v0.1 /bin/bash