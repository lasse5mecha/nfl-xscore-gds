.PHONY: all clean

all:
	pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex

clean:
	rm -f main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.pdf
