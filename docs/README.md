# Documentazione

Raccolta della documentazione del progetto di Intelligenza Artificiale
(telemedicina). La documentazione tecnica del codice vive nel progetto
(`supervised_telemedicina/`); qui trovi il documento didattico sugli aspetti
di IA.

## Contenuto

| Percorso | Cosa |
|----------|------|
| `src/` | Documento LaTeX didattico: "Intelligenza artificiale nel progetto supervised_telemedicina" |
| `sintesi/` | Relazione breve del progetto (sintesi, max 8 pagine) |

## Documento LaTeX (`src/`)

Spiega, partendo dalle basi per chi sa programmare ma non conosce l'IA, tutti
gli aspetti di intelligenza artificiale del progetto: machine learning
supervisionato, dati, reti neurali, backpropagation, valutazione, knowledge
distillation, sicurezza e incertezza, riproducibilità.

Struttura:

```
src/
├── documento.tex          # documento principale (frontespizio, indice, capitoli)
├── preambolo.tex          # pacchetti, stile, comandi custom
├── bibliografia.bib       # riferimenti (Goodfellow, Bishop, Hastie, Hinton)
├── Makefile               # make / make clean / make open
├── capitoli/              # 11 capitoli (01_introduzione … 11_conclusioni)
├── figure/
│   ├── generate/          # 6 PNG generati dai dati reali del progetto
│   └── diagrammi/         # 3 diagrammi TikZ (architettura, gate, distillazione)
└── genera_figure.py       # rigenera le figure dai dati reali
```

Uso:

```bash
cd src
make          # compila documento.pdf (47 pagine)
make open     # compila e apre
```

Le figure PNG sono generate dai dati reali del progetto (stesse funzioni della
dashboard): per rigenerarle dopo un nuovo training, esegui
`python3 genera_figure.py` da `src/`.

## Sintesi-relazione (`sintesi/`)

Relazione breve del progetto (vincolo: max 8 pagine, verificato con
`make check` via `pdfinfo`).

```bash
cd sintesi
make          # compila main.pdf e lo copia in ../sintesi_sistema.pdf (8 pagine)
make check    # compila e verifica Pages <= 8
make clean    # rimuove i file intermedi (resta in docs/sintesi/)
```