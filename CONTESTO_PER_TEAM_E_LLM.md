# AIRIC-dynamic-pricing — contesto per il team e per le LLM

Stato documentato: **18 novembre 2026, kit 0.2.1**. Questo documento raccoglie
il contesto degli ambienti locali, delle cache e delle simulazioni non inclusi
nel repository. È una sintesi verificabile degli artefatti originali, non una
copia delle tracce complete. I conteggi di test e partite qui riportati sono
risultati del controllo già svolto; la redazione di questo documento non è una
nuova esecuzione degli esperimenti.

## 1. Obiettivo e stato del progetto

Prepariamo un agente Python per l'AI Dynamic Pricing Hackathon AIRIC × PMDS:
acquisti di materie prime, produzione e prezzi, con strategie confrontabili e
un adattatore da completare quando saranno disponibili le regole ufficiali.
L'obiettivo è avere una base affidabile e adattabile per competere, senza
confondere le prestazioni nel simulatore con una garanzia di vittoria in gara.

Il progetto funziona da CLI, senza API esterne durante l'esecuzione. Il protocollo
JSON presente è un esempio interno. Non è ancora un'integrazione con il server
ufficiale. I notebook del corso Online Learning, lezioni 03–10, hanno fornito
idee teoriche; non sono dipendenze di runtime e non sono inclusi nel repository.

La baseline era il punto di partenza iniziale. Dopo gli esperimenti locali il
profilo candidato è diventato **`competitive`**, con `exp3` e `gp` come confronti
utili. `adaptive` indica invece SW-UCB sui prezzi, acquisti prudenti e produzione
greedy: è un profilo distinto. La scelta in gara dipenderà da regole e feedback.

Lettura consigliata, con tutti i percorsi relativi alla radice del repository:

| File | Contesto fornito |
|---|---|
| [README.md](README.md) | Avvio, comandi, architettura e strategie |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Ambiente locale e lavoro del team |
| [docs/ADATTAMENTO.md](docs/ADATTAMENTO.md) | Informazioni da acquisire in gara e adattatore |
| [docs/AUDIT.md](docs/AUDIT.md) | Difetti corretti e limiti residui |
| [configs/competitive.toml](configs/competitive.toml) | Configurazione candidata congelata |
| [reports/VERIFICA.md](reports/VERIFICA.md) | Esito dell'ultimo ricontrollo |

## 2. Cosa è escluso e dove resta l'informazione utile

| Artefatti locali esclusi | Contenuto e modo di ricrearli |
|---|---|
| `.venv/`, `venv/` | Interprete e pacchetti installati; ricreare dal `pyproject.toml` |
| `__pycache__/`, `*.pyc` | Bytecode Python; viene rigenerato automaticamente |
| `.pytest_cache/`, cache degli strumenti | Metadati di esecuzione dei controlli; non sono prove di test passati |
| `build/`, `dist/`, `*.egg-info/` | Artefatti di packaging; generati dall'installazione/build |
| `runs/` | Configurazioni risolte, manifest, metriche, tracce e snapshot delle esecuzioni |
| `work/`, `*.log` | Ambienti temporanei, prove intermedie e log locali |

Le **metriche selezionate sono già versionate** in `reports/`: l'esclusione di
`runs/` non significa assenza di risultati. Non sono copiati nel repository i
virtualenv, i file temporanei o le migliaia di tracce delle verifiche precedenti.

Le cache di sistema non vanno confuse con lo **stato funzionale dell'agente**:
decisioni pendenti, feedback già consumati, cache di idempotenza, learner e stato
del generatore casuale fanno parte degli snapshot necessari al ripristino.
Eliminarli a metà partita può cambiare le decisioni o provocare doppi aggiornamenti.

## 3. Ambiente registrato e installazione

Versioni registrate in [reports/verification.json](reports/verification.json):

| Componente | Versione osservata |
|---|---|
| Kit | 0.2.1 |
| Python | 3.12.8 |
| NumPy | 2.2.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.0 |
| pytest | 9.1.1 |
| Sistema | Windows 11, build 10.0.26300 |

Questo elenco documenta l'ambiente del controllo; **non è un lockfile completo**.
I requisiti dichiarati in [pyproject.toml](pyproject.toml) sono Python >=3.12 e
NumPy >=1.26,<3. Gli extra sono `test` (pytest >=8), `optimize` (SciPy >=1.11)
e `plots` (Matplotlib >=3.8). GP-UCB è implementato con NumPy e non richiede SciPy.
SciPy serve alla produzione MILP; Matplotlib è disponibile per analisi facoltative.

Da PowerShell, nella radice del progetto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,optimize]"
.\.venv\Scripts\python.exe -m hackathon_kit validate --config configs/competitive.toml
.\.venv\Scripts\python.exe -m pytest -q
```

Non serve attivare l'ambiente. Per il solo runtime minimo usare `pip install -e .`;
per i test senza SciPy usare `pip install -e ".[test]"`. Su macOS/Linux creare
l'ambiente con `python3 -m venv .venv` e usare `.venv/bin/python`.
L'installazione può scaricare dipendenze; le simulazioni non richiedono rete.

## 4. Regole locali, strategie e invarianti

La configurazione candidata usa 40 turni, liquidità iniziale 150, capacità 5,
due risorse (`metal`, `plastic`) e due prodotti (`widget`, `gadget`). Le ricette
sono rispettivamente 1+1 e 2+1 unità di risorse; i costi di produzione sono 1 e 1.5.
Ogni agente inizia con due unità per risorsa: il patrimonio iniziale è 156.
Prezzi ammessi 5–30, tick 0.01, aste ripetute a unità singola al secondo prezzo,
liquidità rinnovabile dalle vendite. Questi sono **default locali, non regole ufficiali**.

L'ordine è acquisto → produzione → pricing → vendita. Il simulatore prende
decisioni simultanee entro ciascuna fase; scorte persistenti, capacità e liquidità
sono finite. Gli scenari sono `stable`, `shock`, `scarcity`, `aggressive`.
Gli avversari standard usano prezzi fissi, casuali o reattivi ai prezzi precedenti.

Il profilo `competitive` combina modello Poisson approssimato della domanda,
prezzi sensibili alle scorte e piano comune per acquisti/produzione. I parametri
selezionati includono `demand_prior=3.0`, `demand_optimism=0.3`,
`demand_decay=0.97`, `price_elasticity=0.12`, `competition_weight=0.5` e
`planner_limit=256`. Per il resto usare il TOML, non ricostruire i default da questa sintesi.

Proprietà da mantenere quando si modifica il codice:

- Un dato non osservabile resta assente o `None`, non diventa zero.
- `decision_id` collega la scelta al feedback anche se ritardato. Duplicati
  identici sono idempotenti; duplicati contraddittori vengono rifiutati.
- L'adattatore aggrega i feedback parziali prima di `observe`: è ammesso un solo
  feedback finale per decisione.
- Uno stockout informa che la domanda era almeno pari alle vendite; non misura
  la domanda completa. `competitive` usa questa censura nel modello.
- Le ricompense possono essere negative: la normalizzazione usa limiti espliciti;
  valori fuori intervallo vengono segnalati e troncati.
- Ripristino dello stato e rollback delle decisioni fallite conservano anche il RNG.
- Le osservazioni non danno accesso allo stato privato degli avversari o a dati futuri.

## 5. Risultati conservati della versione 0.2.1

Il ricontrollo documentato comprende **215 test passati** nell'ambiente completo,
**213 passati e 2 saltati** nell'ambiente minimo senza SciPy, **580 partite senza
azioni invalide** e **14 partite complete con replay e stato finale corrispondenti**.
I 14 replay coprono i dieci profili, gli altri tre scenari per `competitive` e
un confronto con avversari forti. Tutti usano il seme 500001; sono prove di
integrazione/ripristino, non un campione per scegliere la strategia migliore.

Per le 580 partite, ogni profilo è valutato su 80 partite contro avversari
standard e 60 contro `exp3`, `gp`, `competitive`: 4 × (80+60) = 560.
Altre 20 partite usano quattro copie di `competitive`.

| Profilo | Profitto medio, standard | Quota vittorie, standard | Profitto medio, forti | Quota vittorie, forti |
|---|---:|---:|---:|---:|
| baseline | 16.66 | 1.25% | 3.49 | 0.00% |
| exp3 | 152.55 | 26.25% | 13.23 | 3.33% |
| gp | 194.93 | 36.25% | 24.99 | 6.67% |
| competitive | 311.64 | 46.25% | 228.94 | 46.67% |

La quota di vittorie è `win_share`, con credito frazionario per i pareggi.
Questi aggregati sono descrittivi: i semi si ripetono fra scenari e non vanno
trattati come 580 osservazioni completamente indipendenti.

Dettaglio di `competitive` per mostrare anche le condizioni meno favorevoli:

| Avversari | Scenario | Partite | Profitto medio | Errore standard del profitto | Quota vittorie |
|---|---|---:|---:|---:|---:|
| standard | stable | 20 | 316.91 | 61.17 | 55% |
| standard | shock | 20 | 386.55 | 55.39 | 65% |
| standard | scarcity | 20 | 401.00 | 88.41 | 45% |
| standard | aggressive | 20 | 142.10 | 33.28 | 20% |
| forti | stable | 20 | 219.52 | 31.25 | 40% |
| forti | shock | 20 | 212.84 | 21.95 | 40% |
| forti | scarcity | 20 | 254.47 | 35.49 | 60% |
| copie di competitive | stable | 20 | 555.04 | 24.17 | 20% |

Lo scenario `aggressive` cambia gli avversari standard: è escluso dal gruppo
forte per non duplicare `stable`. Le prestazioni contro copie dello stesso
profilo non sono direttamente confrontabili con quelle contro altri gruppi.
Nel simulatore il profitto è margine realizzato meno magazzino; il patrimonio
include le scorte al costo. Guardare anche liquidità, invenduto e ranghi.

Fonti già disponibili a chi clona o scarica il repository:

| File | Evidenza |
|---|---|
| [reports/verification.json](reports/verification.json) | Versioni, conteggi, aggregati e risultati dei replay |
| [reports/recheck-manifest.json](reports/recheck-manifest.json) | Semi, hash, configurazioni e avversari effettivi; stato `complete` |
| [reports/recheck-runs.csv](reports/recheck-runs.csv) | 580 righe con profitto, patrimonio, liquidità minima/finale, scorte, vendite, costi, ranghi e tempi |
| [reports/recheck-summary.csv](reports/recheck-summary.csv) | Medie ed errore standard per profilo, gruppo e scenario |
| [reports/recheck-paired.csv](reports/recheck-paired.csv) | Differenze di profitto appaiate per seme rispetto a baseline, EXP3 e GP |

`reports/legacy-v0.1/` e `reports/legacy-v0.2/` contengono risultati storici.
Non mescolarli alla versione corrente: correzioni del simulatore o degli agenti
possono cambiare il significato dei numeri.

## 6. Semi, selezione e riproducibilità

**La valutazione 0.2.1 è una regressione, non una nuova valutazione indipendente.**
Riusa i semi 300000–300019 già impiegati nello studio 0.2; il report dichiara
`fresh_independent_evaluation=false`. Non sono stati riottimizzati i parametri
durante questo controllo.

Esistono due protocolli diversi:

| Protocollo | Sviluppo | Valutazione | Criterio/uso |
|---|---|---|---|
| CLI `benchmark` predefinito | 0–19 | 100000–100019 | 640 partite: 4 profili × 4 scenari × 40 semi |
| `scripts/competition_study.py` | 200–203 | 300000–300019 | 112 partite di selezione, poi 580 di valutazione |

Lo studio confronta quattro combinazioni di `demand_prior` (1.5/3.0) e
`demand_optimism` (0.0/0.3). Seleziona la quota di vittorie media più alta,
con profitto medio come spareggio. Il comando CLI `tune` seleziona invece
per profitto medio: non confondere i due criteri.

**Lo script di studio ha semi fissi nel codice e non espone flag per cambiarli.**
Rilanciarlo in una directory nuova ripete lo stesso protocollo. Per valutazioni
su nuovi intervalli usare i flag del benchmark, oppure modificare e congelare
esplicitamente il protocollo dello studio prima della selezione. Se i risultati
di valutazione influenzano una modifica, usare successivamente un altro insieme
di semi e annotare quali intervalli sono già stati osservati.

Impronte della build descritta, tratte dal manifest:

```text
code_sha256:
6361734b706c27384f047e57304b4561ebf239e5edd57882581153c9539ad7f0
study_sha256:
84f0a3bdaf565ca2d86091373feba3c6e81e59fe3b1e4c99b4778a8d9d767c48
```

`source_fingerprint()` concatena nome e byte dei moduli `hackathon_kit/*.py`
in ordine e calcola SHA-256; non è l'hash dell'intero repository. Il manifest
registra separatamente configurazioni e hash dello script di studio/selezione.
Cambiare solo documentazione non modifica l'impronta del codice. Le eccezioni
di fine riga in `.gitattributes` per `learners.py` e `competitive.toml` conservano
i byte verificati: anche una conversione CRLF/LF può cambiare un hash.

Lo schema degli snapshot agente è **2**; lo schema delle tracce JSONL correnti
è **1**. Sono formati diversi. Gli snapshot del kit 0.1 non vengono migrati
automaticamente. Per riprodurre vecchie tracce serve il codice del loro manifest.
I tempi di decisione dipendono da macchina e carico; non devono coincidere
numericamente fra due esecuzioni.

## 7. Inventario delle esecuzioni originali escluse

Nella copia di lavoro originaria `runs/` conteneva **3.554 file, 83.530.147 byte**
(circa 83,53 MB decimali). Questa è una fotografia della preparazione, non una
cartella che deve esistere nel clone dei colleghi.

| Directory originale sotto `runs/` | File | Byte | Contesto |
|---|---:|---:|---|
| `recheck-2026-11-18/` | 5 | 614.036 | Studio 0.2.1: manifest, CSV e checkpoint parziali |
| `recheck-traces-2026-11-18/` | 70 | 3.684.703 | 14 partite con configurazione, manifest, metriche, stato e traccia |
| `audit-development/` | 4 | 36.735 | Selezione precedente dei parametri |
| `audit-holdout/` | 4 | 126.742 | Valutazione precedente |
| `audit-holdout-final/` | 4 | 126.651 | Valutazione precedente consolidata |
| `audit-release-traces/` | 50 | 2.427.147 | Tracce di release precedenti |
| `audit-trace/` | 5 | 268.829 | Traccia di audit precedente |
| `benchmark-release/` | 3.203 | 71.436.897 | Benchmark completo precedente e relative tracce |
| `first-price-verified/` | 43 | 1.147.808 | Verifiche precedenti delle aste al primo prezzo |
| `optional-profiles/` | 123 | 2.729.707 | Verifiche precedenti dei profili opzionali |
| `pacing-verified/` | 43 | 930.892 | Verifiche precedenti del pacing |

I nomi di queste directory documentano provenienza, non equivalenza fra versioni.
I CSV/manifest selezionati dello studio attuale sono copiati in `reports/`.
Lo studio massivo registra metriche per partita, ma **non le tracce complete
di tutte le sue 580 partite**. Le 14 tracce di ricontrollo sono esecuzioni distinte.

Una partita creata con `simulate` salva:

- `config.toml`: configurazione effettiva risolta;
- `manifest.json`: versioni, seme, hash del codice, avversari e stato del run;
- `metrics.csv`: una riga di metriche finali;
- `state.json`: snapshot finale dell'agente valutato;
- `trace.jsonl.gz`: intestazione, osservazioni/azioni, feedback e stato finale.

Una traccia permette di ricostruire le scelte dell'agente; non è una registrazione
onnisciente dello stato privato di tutti gli avversari. `replay` confronta decisioni
e stato, non ricalcola i tempi di esecuzione. `partial-runs.jsonl` dello studio è
un checkpoint di metriche già completate, non una traccia compatibile con `replay`.

## 8. Comandi per ricreare risultati e tracce

Eseguire dalla radice; nei comandi seguenti `python` è quello dell'ambiente
installato. Su Windows si può sostituirlo con `.\.venv\Scripts\python.exe`.
Ogni `--out` deve indicare una directory nuova: i comandi rifiutano sovrascritture.

Partita della configurazione candidata e replay:

```powershell
python -m hackathon_kit simulate --config configs/competitive.toml --seed 500001 --scenario stable --out runs/team-competitive
python -m hackathon_kit replay runs/team-competitive/trace.jsonl.gz
```

Confronto con avversari forti:

```powershell
python -m hackathon_kit simulate --config configs/competitive.toml --seed 500001 --opponents exp3 gp competitive --out runs/team-crossplay
```

Ripetizione dello studio di valutazione con la configurazione già congelata
(580 partite, stessi semi noti, senza rifare la selezione):

```powershell
python scripts/competition_study.py evaluation --selection configs/competitive.toml --out runs/team-recheck --workers 4
```

Ripetizione del protocollo di selezione e valutazione, sempre sui semi fissi:

```powershell
python scripts/competition_study.py development --out runs/team-development --workers 4
python scripts/competition_study.py evaluation --selection runs/team-development/selected.toml --out runs/team-evaluation --workers 4
```

Esempio di benchmark con intervalli diversi da quelli documentati sopra
(verificare con il team che non siano già stati usati; questo comando non è
stato eseguito per redigere il documento):

```powershell
python -m hackathon_kit benchmark --config configs/competitive.toml --profiles baseline exp3 gp competitive --opponents exp3 gp competitive --scenarios stable shock scarcity --dev-start 700000 --eval-start 800000 --dev-seeds 20 --eval-seeds 20 --out runs/team-new-seeds
```

Il benchmark produce anche tracce e può occupare molto più spazio dello studio.
Conservare comandi, configurazione, hash, semi e report insieme. Per condividere
nuovi risultati selezionati, copiarli in una sottocartella distinta di `reports/`
con provenienza e data, mantenendo separati i report storici.

## 9. Limiti da conoscere prima di proporre miglioramenti

- La domanda del modello è un'approssimazione; il simulatore non riassegna la
  domanda insoddisfatta ad altri venditori dopo gli stockout.
- L'invenduto è valorizzato al costo; i costi di magazzino sono addebitati entro
  la liquidità disponibile. Punteggio o insolvenza ufficiali possono essere diversi.
- Il piano è su un turno, con scorte obiettivo. Il beam limita il costo di ricerca
  ma può perdere l'ottimo; il piano espone `exact` e `pruned_states`.
- Inventari molto più grandi, molti prodotti o episodi lunghi richiedono nuove
  misure di tempi/memoria. Le cache di idempotenza crescono con la partita.
- Il simulatore risolve subito le aste. Con feedback differiti occorre gestire
  liquidità impegnata e disponibilità effettiva nell'adattatore.
- `competitive` richiede vendite e censura per prodotto. Conversion TS richiede
  successi/tentativi osservabili; non è Thompson Sampling sul profitto.
- Pacing richiede asta al secondo prezzo e budget totale fisso sugli acquisti.
  Non trasferire la formula a budget alimentati dalle vendite senza adattarla.
- Le garanzie dei bandit dei notebook non valgono automaticamente per l'intero
  gioco con inventario, produzione e avversari adattivi.
- I 215 test e le 580 partite sono evidenza di controllo, non una prova di assenza
  di difetti né di superiorità in ogni scenario. L'audit descrive le correzioni.

## 10. Come usare questo contesto con una LLM

Fornire il repository insieme a questo documento e specificare lo scopo:
revisione dei contratti, miglioramento delle strategie, nuova valutazione oppure
collegamento all'ambiente ufficiale. Per analisi dei risultati bastano spesso i
CSV di `reports/`; per diagnosticare una decisione precisa serve invece una
traccia riprodotta o l'artefatto originale corrispondente.

Distinguere sempre: fatti verificabili nel codice, risultati registrati nei
report, ipotesi del simulatore e proposte ancora da testare. Non attribuire alle
regole ufficiali i default locali e non inventare il contenuto dei file esclusi.
La priorità al briefing è registrare punteggio, fasi, aste, osservabilità e limiti
di esecuzione; quindi collegare l'adattatore, completare una partita con la
baseline e solo dopo confrontare le strategie compatibili.
