lint:
	python -m pylint --rcfile=.pylintrc toppra
	pycodestyle toppra --max-line-length=120 --ignore=E731,W503
	pydocstyle toppra

DOC:
	echo "Buidling toppra docs"
	sphinx-build -b html docs/source docs/build

gen-tags:
	ctags -Re --exclude='.tox' --exclude='venv' --exclude='docs' --exclude=='dist' .

coverage: 
	python -m pytest -q --cov-report term --cov-report xml --cov=toppra tests


# todos before publishing:
# - increment version in setup.py
# - increment version in doc
# - create a new release to master
# - run publish on master
publish:
	pip install 'twine>=1.5.0'
	python setup.py sdist
	twine upload dist/*
	rm -fr build dist .egg requests.egg-info
