lint:
	python -m pylint --rcfile=.pylintrc toppra
	pycodestyle toppra --max-line-length=120 --ignore=E731,W503
	pydocstyle toppra

docs:
	@echo "Buidling toppra docs"
	@sphinx-build -b html docs/source docs/build
