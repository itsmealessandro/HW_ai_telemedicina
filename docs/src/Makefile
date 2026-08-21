# Makefile --- build del documento LaTeX.
#
# Compila dentro docs/src/ e copia il PDF in docs/ (sovrascrivendo
# documento.pdf), così l'albero è:
#   docs/documento.pdf        <- PDF compilato
#   docs/src/                 <- sorgenti (tex, bib, capitoli, figure, tools)
#
# Uso:
#   make            compila il PDF e lo mette in ../documento.pdf
#   make clean      rimuove i file intermedi (restano in docs/src/)
#   make open       compila e apre il PDF in docs/

MAIN    = documento
OUTDIR  = ..
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX  = bibtex

.PHONY: all clean open

all: $(OUTDIR)/$(MAIN).pdf

$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex preambolo.tex capitoli/*.tex bib/bibliografia.bib
	$(PDFLATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN) || true
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex
	cp $(MAIN).pdf $(OUTDIR)/$(MAIN).pdf

open: all
	xdg-open $(OUTDIR)/$(MAIN).pdf

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).pdf
