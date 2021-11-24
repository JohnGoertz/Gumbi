SOURCEDIR := $(pwd)
notebooks_pypercent := $(shell find $(SOURCEDIR) -name '*.pct.py')
notebooks_markdown := $(shell find $(SOURCEDIR) -name '*.md')

notebooks_ipynb_py = $(notebooks_pypercent:.pct.py=.ipynb)
notebooks_ipynb_md = $(notebooks_markdown:.md=.ipynb)

.SUFFIXES: .ipynb .pct.py .md

.PHONY: all
all: $(notebooks_ipynb_md) $(notebooks_ipynb_py)

$(notebooks_ipynb_py): %.ipynb: %.pct.py
	jupytext --execute --to notebook -o $@ $<

$(notebooks_ipynb_md): %.ipynb: %.md
	jupytext --execute --to notebook -o $@ $<

.PHONY: pair-ipynb
pair-ipynb:
	jupytext --set-formats ipynb,md installation.ipynb
	jupytext --set-formats ipynb,.pct.py:percent design/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent build/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent test/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent learn/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent lims/*.ipynb