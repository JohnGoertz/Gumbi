# Gumbi documentation

## Compile documentation locally
This will generate .html within `/docs/build/html/` that you can open and browse like a real webpage.

### Setup
Install doc dependencies
```
conda install sphinx sphinx_rtd_theme numpydoc nbsphinx ipython jupytext jupyter_client
```

### Compile
From within the `/docs/` directory:

1. Convert and execute notebooks (.pct.py -> .ipynb)
    * `make -C source/notebooks`

2. Autodocument the API (.py -> .rst)
    * `python source/generate_api_rst.py`

3. Compile
    * `make clean && make html`
