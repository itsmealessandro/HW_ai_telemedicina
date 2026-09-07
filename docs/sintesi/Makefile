# Makefile — relazione breve (docs/sintesi/)
#
#   make          compila il PDF in ../sintesi_sistema.pdf
#   make clean    rimuove i file intermedi (resta in docs/sintesi/)
#   make check    compila e verifica pdfinfo Pages <= 8

MAIN    = main
OUTPDF  = ../sintesi_sistema.pdf
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean check

all: $(OUTPDF)

$(OUTPDF): $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex
	cp $(MAIN).pdf $(OUTPDF)

check: all
	pdfinfo $(OUTPDF) | grep Pages
	test $$(pdfinfo $(OUTPDF) | awk '/Pages:/ {print $$2}') -le 8

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
