# Lavorare sul progetto

Python 3.12 o successivo. Eseguire i comandi dalla radice del repository.

## Installazione

Windows / PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,optimize]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m hackathon_kit simulate --profile competitive --out runs/prova
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test,optimize]"
.venv/bin/python -m pytest -q
.venv/bin/python -m hackathon_kit simulate --profile competitive --out runs/prova
```

## Primo push su GitHub

Creare su GitHub un repository vuoto, senza inizializzare file sul sito:
README e gitignore sono già presenti localmente. Sostituire l'URL sotto con
quello del repository creato:

```bash
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin HEAD
```

Il comando pubblica il branch locale corrente. Se manca ancora il commit iniziale:

```bash
git add .
git commit -m "Initial hackathon kit v0.2.1"
```

## Flusso per il team

Clonare il repository, creare un branch per ogni modifica e aprire una pull request:

```bash
git pull --ff-only
git switch -c codex/nome-modifica
# modificare il codice ed eseguire i test
git add percorso/dei/file
git commit -m "Descrizione della modifica"
git push -u origin HEAD
```

Prima di unire una modifica: test passati, configurazione riproducibile e verifica
di una simulazione quando cambia il comportamento dell'agente. Coordinare fra
colleghi le modifiche allo stesso modulo.

- Adattatore e contratti ufficiali: `hackathon_kit/contracts.py` e `docs/ADATTAMENTO.md`.
- Coordinamento e strategie: `agent.py`, `planning.py`, `learners.py`.
- Ambiente e confronti: `simulator.py`, `scripts/competition_study.py`.
- Configurazioni: `configs/`; partire da `competitive.toml`.

## File da versionare

Codice, test, configurazioni condivise, documentazione e report selezionati.
`runs/`, `.venv/`, cache, build e file `.env` sono esclusi. Non forzare il commit
di questi file. `.gitattributes` mantiene LF nei sorgenti e nei TOML, con due
eccezioni esplicite per conservare i byte originali dei file già verificati:
`hackathon_kit/learners.py` e `configs/competitive.toml`. Gli esperimenti registrano
gli hash dei file; evitare conversioni automatiche dei fine riga su questi file.

I report presenti descrivono la versione testata; dopo cambiamenti comportamentali
rigenerare confronti e report, senza riutilizzare statistiche come se fossero nuove.
