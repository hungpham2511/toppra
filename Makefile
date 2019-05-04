lint:
	python -m pylint --rcfile=.pylintrc toppra
	pycodestyle toppra --max-line-length=120 --ignore=E731,W503
	pydocstyle toppra

doc: 
	echo "Buidling toppra docs"
	sphinx-build -b html docs/source docs/build

coverage: 
	python -m pytest -q --cov-report term --cov-report xml --cov=toppra tests

publish:
	pip install 'twine>=1.5.0'
	python setup.py sdist
	twine upload dist/*
	rm -fr build dist .egg requests.egg-info
