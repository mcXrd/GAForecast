test:
	pytest

# Run mypy for type checking
mypy:
	mypy models

# Format code with black
black:
	black models

# Format imports with isort
isort:
	isort models

# Run black and isort for linting
lint: black isort
