# AIRIC-dynamic-pricing

Repository del team per AIRIC × PMDS. Per lavorare insieme e per il primo push,
vedere [CONTRIBUTING.md](CONTRIBUTING.md).

Per il contesto degli ambienti locali, degli esperimenti e degli artefatti esclusi,
leggere [CONTESTO_PER_TEAM_E_LLM.md](CONTESTO_PER_TEAM_E_LLM.md).

**Versione 0.2.1, ricontrollata il 18 novembre 2026: partire dal profilo `competitive`**, mantenendo `exp3` e `gp` come
alternative da confrontare quando arriveranno le regole. Il nuovo profilo coordina
domanda, scorte, acquisti e produzione; la configurazione selezionata su dati di
sviluppo è `configs/competitive.toml`.

```powershell
python -m hackathon_kit simulate --config configs/competitive.toml --out runs/competitive-first
python -m hackathon_kit simulate --profile competitive --opponents exp3 gp competitive --out runs/crossplay-first
```

Il report aggiornato è [VERIFICA.md](reports/VERIFICA.md); l'audit è in [AUDIT.md](docs/AUDIT.md).
Snapshot versione 1 e vecchie tracce richiedono il pacchetto 0.1, non incluso in questo repository:
le versioni 0.2 e 0.2.1 usano snapshot versione 2 e rifiutano un ripristino incompatibile.
La 0.2.1 valida anche la coerenza fra decisioni dell'agente e learner. Per riprodurre
esattamente una vecchia traccia usare la versione del codice indicata nel suo manifest.

Kit Python 3.12 interamente offline durante l'esecuzione. Include strategie di pricing,
acquisto e produzione, simulatore, benchmark, tuning e riproduzione delle decisioni.
Le regole del simulatore sono ipotesi didattiche: non descrivono il regolamento ufficiale.

## Avvio su Windows / PowerShell

Dalla cartella di questo progetto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,optimize]"
.\.venv\Scripts\python.exe -m hackathon_kit validate
.\.venv\Scripts\python.exe -m hackathon_kit simulate --profile baseline --out runs/first
.\.venv\Scripts\python.exe -m hackathon_kit replay runs/first/trace.jsonl.gz
.\.venv\Scripts\python.exe -m pytest -q
```

Non serve attivare il virtualenv. Installazione minima: `pip install -e .` (solo NumPy).
L'extra `optimize` abilita SciPy/MILP, `test` abilita pytest e `plots` rende disponibile
Matplotlib per analisi personalizzate. Dopo l'installazione esiste anche il comando
`hackathon`. Nei comandi seguenti `python` indica il Python dell'ambiente scelto.
Se NumPy è già disponibile, `python -m hackathon_kit` funziona direttamente dalla cartella.

## Comandi

```powershell
python -m hackathon_kit simulate --profile adaptive --scenario shock --seed 7 --out runs/adaptive-7
python -m hackathon_kit benchmark --out runs/comparison
python -m hackathon_kit tune --out runs/tuning
python -m hackathon_kit validate --config configs/first-price.toml
python -m hackathon_kit simulate --config configs/first-price.toml --out runs/first-price
python -m hackathon_kit simulate --config configs/pacing.toml --out runs/pacing
python -m hackathon_kit simulate --profile gp --out runs/gp
python -m hackathon_kit simulate --profile conversion --out runs/conversion
python -m hackathon_kit simulate --profile milp --out runs/milp
python -m hackathon_kit act --input examples/observation.json --state runs/agent.json
```

Ogni cartella di esecuzione deve essere nuova: il programma rifiuta di sovrascriverla.
`act` aggiorna invece intenzionalmente il file di stato indicato, tramite sostituzione atomica.
Non usare lo stesso file di stato da processi concorrenti.

Benchmark rapido:

```powershell
python -m hackathon_kit benchmark --profiles baseline adaptive --turns 8 --dev-seeds 2 --eval-seeds 2 --out runs/quick
```

Il benchmark completo predefinito esegue **640 partite**: quattro profili (`baseline`,
`adaptive`, `ucb`, `exp3`), quattro scenari, 20 semi di sviluppo e 20 di valutazione.
I semi sono rispettivamente 0–19 e 100000–100019. I flussi casuali di domanda/aste
sono separati da quelli degli agenti. I parametri degli avversari restano fissi
quando si modifica l'agente valutato. GP, conversione, MILP e le strategie d'asta
si includono esplicitamente tramite `--profiles`, con una configurazione compatibile.

Il tuning confronta SW-UCB con finestre 10/20/40 e frazioni di liquidità 0.25/0.4
su tutti gli scenari e solo i semi di sviluppo. `selected.toml` è una configurazione
completa scelta per profitto medio, non per la migliore singola partita.
Per valutarla su dati separati:

```powershell
python -m hackathon_kit benchmark --config runs/tuning/selected.toml --profiles adaptive baseline --out runs/held-out
```

Consultare le righe `evaluation` del report. Il comando esegue anche lo split di
sviluppo per confronto; non modifica il modello selezionato. I profili da CLI
sostituiscono insieme pricing, acquisti e produzione, lasciando gli altri parametri
alla configurazione per evitare combinazioni incompatibili.

## Risultati e interpretazione

Ogni partita salva `config.toml` effettivo, `manifest.json` con seme/versioni,
`metrics.csv`, `state.json` e `trace.jsonl.gz`. La traccia contiene osservazioni,
azioni, feedback e stato finale del solo agente valutato. `replay` ricostruisce
azioni e stato e fallisce alla prima differenza: non ricalcola i tempi di esecuzione.

Il benchmark salva `runs.csv` e `summary.csv`: medie, deviazioni standard,
errore standard del profitto e decimo percentile, separati per profilo/scenario/split.
Tempi medi e massimi sono misure hardware dipendenti, escluse dalle verifiche di
riproducibilità. Nessuna classifica viene dedotta dalla migliore partita.

Nel simulatore:
- profitto = margine realizzato sulle vendite meno costi di magazzino;
- patrimonio = liquidità + scorte valutate al costo medio di acquisto/produzione;
- patrimonio finale = patrimonio iniziale + profitto;
- ogni agente parte con 150 di liquidità e due unità per materia prima (patrimonio
  iniziale 156): una scorta minima permette di sperimentare il pricing fin dall'inizio;
- gli acquisti e la produzione trasferiscono valore dalla liquidità alle scorte;
- i costi di magazzino sono limitati alla liquidità disponibile (regola didattica);
- invenduto valutato al costo non equivale a ricavo garantito: guardare anche liquidità e scorte.

L'esecuzione fallisce su un'azione invalida. `invalid_actions = 0` in un report
completo significa che tutti i controlli sono passati; un'esecuzione interrotta
lascia una traccia incompleta che `replay` rifiuta.

## Moduli e contratti

| Modulo | Responsabilità |
|---|---|
| `contracts.py` | Dataclass, protocolli e adattatore JSON di esempio; controlli sulle azioni |
| `agent.py` | Coordinamento delle fasi, inventario osservato, selezione strategie, stato |
| `learners.py` | UCB, SW-UCB, EXP3-IX, GP-UCB e Thompson Sampling per conversioni |
| `planning.py` | Domanda censurata e pianificazione congiunta delle scorte |
| `prices.py` | Arrotondamento e controlli decimali dei tick, anche inferiori a 1e-10 |
| `simulator.py` | Mercato locale, aste, domanda e contabilità indipendenti dagli agenti |
| `config.py` | Default dichiarati, profili, lettura TOML e compatibilità |
| `cli.py` | Comandi, artefatti, benchmark, tuning e replay |

L'agente implementa `reset`, `act`, `observe`, `state_dict`, `load_state_dict`.
Un `decision_id` identifica una decisione per tutta la partita. Ripetere `act`
con lo stesso identificativo e la stessa osservazione restituisce l'azione originale.
Un identificativo riutilizzato con dati diversi viene rifiutato. Un solo feedback
finale è ammesso per decisione; duplicati identici sono ignorati, duplicati contraddittori,
identificativi sconosciuti o fasi errate sono rifiutati. Se il server fornisce feedback parziali, l'adattatore
deve aggregarli prima di chiamare `observe`.

Stato e probabilità originarie delle scelte sono conservati per feedback differiti.
I campi non osservabili restano `None` o assenti nelle mappe: dati richiesti da una
strategia provocano un errore se mancanti. Il simulatore espone solo scorte proprie,
costi propri, limiti e media pubblica dei prezzi concorrenti del turno precedente.

## Strategie e ipotesi

- **Competitive:** distribuzione di domanda Poisson con effetto del prezzo proprio
  e della media pubblica dei prezzi concorrenti. Aggiornamento bayesiano su 25 ipotesi,
  con oblio; uno stockout usa la probabilità `P(domanda >= vendite)`. La quantità
  ottima considera la probabilità di vendere le scorte; gli acquisti dipendono dal
  piano completo, dai componenti mancanti e dalla liquidità da riservare per produrre.
  Ogni asta riserva anche il costo delle altre unità dello stesso input ancora necessarie.
  La ricerca è esatta per i piccoli casi predefiniti, limitata tramite beam search
  nei casi più grandi; il piano espone `exact` e `pruned_states`. Il modello è un'approssimazione dichiarata e non conosce le
  formule nascoste del simulatore.

- **Baseline:** prezzo a costo stimato più margine; acquisti limitati da valore
  produttivo, liquidità e fabbisogno; produzione greedy con limite di scorte obiettivo.
- **UCB/SW-UCB:** sette prezzi iniziali; SW-UCB conserva gli ultimi 20 indici di
  decisione di pricing per prodotto. Una ricompensa tardiva fuori finestra è ignorata.
  Esplorazione e pareggi sono randomizzati in modo riproducibile; le decisioni
  ancora senza feedback sono conteggiate per non esplorare sempre lo stesso prezzo.
- **EXP3-IX:** perdite trasformate in [0,1], probabilità della scelta originale e
  aggiornamenti in logaritmi per stabilità numerica.
- **GP-UCB:** kernel RBF su prezzi equispaziati, rumore regolarizzato e massimo 128
  osservazioni. Implementazione NumPy, nessun servizio esterno.
- **Conversion TS:** campioni Beta per conversioni e selezione per margine unitario
  atteso. Richiede successi/tentativi osservabili; salta dati censurati. Saltare gli
  stockout non elimina ogni possibile bias di selezione: profilo sperimentale.
- **Aste al primo prezzo:** EXP3-IX sceglie fra cinque frazioni dell'offerta massima
  prudente. La ricompensa è un proxy del valore produttivo meno il pagamento, non
  il profitto definitivo dell'azienda.
- **Pacing:** solo aste al secondo prezzo e budget totale fisso sugli acquisti;
  il simulatore impone il budget anche quando le vendite generano nuova liquidità.
- **MILP:** massimizza margini stimati sotto vincoli di risorse/capacità/liquidità;
  limite di un secondo e ripiego greedy se la soluzione manca o non è ammissibile.

I learner di pricing ottimizzano margini realizzati per prodotto, che dipendono anche
dalle scorte e dalle scelte precedenti. Le garanzie teoriche dei notebook non si
trasferiscono automaticamente a questo gioco con stato e concorrenti adattivi.
Le ricompense sono normalizzate con limiti espliciti: fuori intervallo si emette un
avviso e si tronca il valore. I default ampi sono conservativi; regolarli dopo aver
capito le unità del gioco. Lo stockout non viene presentato come domanda completa.

La pipeline assume acquisto → produzione → prezzo → vendita. Aste multiunità,
ricette diverse, domanda osservabile, produzione differita o vincoli ulteriori
richiedono modifiche esplicite. Consultare [ADATTAMENTO.md](docs/ADATTAMENTO.md).

Le mappe TOML `reserve`, `supply`, `initial_raw`, `recipes` e `production_cost`
sostituiscono interamente le rispettive mappe predefinite: configurarle insieme
quando si cambiano risorse o prodotti. Gli altri parametri omessi mantengono il default.

I risultati verificati della versione consegnata sono in [VERIFICA.md](reports/VERIFICA.md).
Il repository include i report selezionati. Le tracce complete e gli snapshot
si generano localmente in `runs/`, esclusa da Git insieme ad ambienti e cache.

Per ripetere lo studio con avversari forti sui semi fissi del protocollo:

```powershell
python scripts/competition_study.py development --out runs/my-development --workers 4
python scripts/competition_study.py evaluation --selection runs/my-development/selected.toml --out runs/my-holdout --workers 4
```

Il protocollo esegue 112 partite di selezione, poi 580 partite con parametri fissi:
avversari predefiniti, gruppo EXP3/GP/competitive e copie del profilo selezionato.
Gli sweep salvano metriche per partita, configurazioni, semi e hash del codice;
registrano anche l'hash dello script, i roster effettivi e i risultati parziali
in caso di errore. Tutti i profili di valutazione usano il mercato della configurazione selezionata.
`simulate` e `benchmark` salvano anche le tracce complete. Lo scenario aggressivo
è escluso dal gruppo forte perché modifica solo gli avversari predefiniti e
altrimenti duplicherebbe lo scenario stabile.

Il tuning CLI supporta griglie esplicite su qualsiasi opzione agente:

```powershell
python -m hackathon_kit tune --profile competitive --parameter 'demand_decay=[0.9,0.97]' --opponents exp3 gp competitive --out runs/custom-tuning
```

`benchmark` accetta `--dev-start` e `--eval-start` per intervalli nuovi, ne verifica
la separazione e produce `paired.csv` con differenze su semi corrispondenti.
Se si modificano parametri dopo aver osservato la valutazione, serve un nuovo
insieme di verifica.

## Test

```powershell
python -m pytest -q
```

I test coprono contabilità, aste e pareggi, limiti fisici/monetari, scenari,
ricompense negative, censura, feedback differiti/duplicati, ripristino del generatore
casuale, replay, CLI e separazione dei semi. Se un ambiente protetto impedisce
l'uso della cartella temporanea di sistema, passare `--basetemp` con un percorso
dedicato e scrivibile: pytest può svuotarlo prima di usarlo.

I notebook del corso hanno fornito il riferimento concettuale (lezioni 03–10).
Le implementazioni qui sono autonome; non eseguono né modificano i notebook.

Il nuovo controllo del 18 novembre comprende ripristini malformati, rollback delle
decisioni fallite, JSON con chiavi duplicate, domanda Poisson elevata, tick piccoli,
capacità nominali molto grandi e immutabilità delle tracce. Vedere il report per
i conteggi aggiornati; i risultati delle versioni precedenti sono archiviati.
