all: clean run

clean:
	@echo "Cleaning up..."
	@rm -rf storage/
	@rm -rf chroma/

run:
	@echo "Running..."
	@python starter.py

lint:
	@echo "Linting..."
	@pre-commit run --all-files
