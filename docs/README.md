# Documentazione

Raccolta della documentazione del progetto di Intelligenza Artificiale
(telemedicina). La documentazione tecnica del codice vive nel progetto
(`supervised_telemedicina/`); qui trovi il documento didattico sugli aspetti
di IA.

## Contenuto

| Percorso | Cosa |
|----------|------|
| `latex/` | Documento LaTeX didattico: "Intelligenza artificiale nel progetto supervised_telemedicina" |

## Documento LaTeX (`latex/`)

Spiega, partendo dalle basi per chi sa programmare ma non conosce l'IA, tutti
gli aspetti di intelligenza artificiale del progetto: machine learning
supervisionato, dati, reti neurali, backpropagation, valutazione, knowledge
distillation, sicurezza e incertezza, riproducibilità.

Struttura:

```
latex/
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
cd latex
make          # compila documento.pdf (45 pagine)
make open     # compila e apre
```

Le figure PNG sono generate dai dati reali del progetto (stesse funzioni della
dashboard): per rigenerarle dopo un nuovo training, esegui
`python3 genera_figure.py` da `latex/`.