install:
	pip install -r requirements.txt 
    
plotly-extension:
	jupyter labextension install jupyterlab-plotly@4.9.0

gitignore:
	touch .gitignore; echo ".gitignore" >> .gitignore
    
setup: install plotly-extension gitignore

nbstripout:
	nbstripout *.ipynb

black:
	black *.py
