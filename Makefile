.PHONY: help init add-data push pull status clean run-pipeline compile-pipeline

# Variabili
DATA_DIR := data/documents
PIPELINE_FILE := document_pipeline.yaml

help:
	@echo "Comandi disponibili:"
	@echo "  make init          - Inizializza DVC e crea struttura directories"
	@echo "  make add-data      - Traccia nuovi documenti con DVC"
	@echo "  make push          - Push dati su MinIO"
	@echo "  make pull          - Pull dati da MinIO"
	@echo "  make status        - Mostra stato DVC"
	@echo "  make update        - Add + Push (workflow completo)"
	@echo "  make compile-pipeline - Compila pipeline Kubeflow"
	@echo "  make run-pipeline  - Esegui pipeline Kubeflow"
	@echo "  make clean         - Rimuove file temporanei"

init:
	@echo "Inizializzazione DVC..."
	@if [ ! -d ".dvc" ]; then \
		dvc init; \
		echo "DVC inizializzato"; \
	else \
		echo "DVC già inizializzato"; \
	fi
	@mkdir -p $(DATA_DIR)
	@echo "Directory $(DATA_DIR) creata"

add-data:
	@echo "Tracciamento documenti con DVC..."
	@dvc add $(DATA_DIR)
	@git add $(DATA_DIR).dvc .gitignore
	@echo "Documenti tracciati. Ricorda di committare le modifiche con git."

push:
	@echo "Push dati su MinIO..."
	@dvc push
	@echo "Push completato"

pull:
	@echo "Pull dati da MinIO..."
	@dvc pull
	@echo "Pull completato"

status:
	@echo "Status DVC:"
	@dvc status

update: add-data push
	@echo "✓ Dati aggiunti e caricati su MinIO"
	@echo "Esegui 'git commit -m \"Update data\"' per salvare le modifiche"

compile-pipeline:
	@echo "Compilazione pipeline Kubeflow..."
	@python kubeflow_pipeline.py
	@echo "Pipeline compilata in $(PIPELINE_FILE)"

run-pipeline: compile-pipeline
	@echo "Esecuzione pipeline Kubeflow..."
	@python run_pipeline.py

clean:
	@echo "Pulizia file temporanei..."
	@rm -f $(PIPELINE_FILE)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Pulizia completata"