# Makefile — build del documento LaTeX (senza latexmk: sequenza esplicita).
#
# Uso:
#   make            compila il PDF (pdflatex + bibtex + 2 passate)
#   make clean      rimuove i file intermedi
#   make open       compila e apre il PDF

MAIN = documento
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

.PHONY: all clean open

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex preambolo.tex capitoli/*.tex bibliografia.bib
	$(PDFLATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN) || true
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex

open: $(MAIN).pdf
	xdg-open $(MAIN).pdf

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).pdf