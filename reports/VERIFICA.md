# Ricontrollo del 18 novembre 2026 — versione 0.2.1

## Esito

- La suite precedente passava **157 test**; le regressioni aggiunte portano il
  totale a **215 test passati** nell'ambiente completo.
- Nel virtualenv minimo: **213 passati e 2 saltati**, perché i due controlli
  opzionali richiedono SciPy. Installazione 0.2.1, comando CLI e replay verificati
  anche da fuori dalla cartella sorgente.
- **580 partite di ricontrollo**, tutte completate senza azioni invalide; verificate
  contabilità, configurazioni e corrispondenza fra hash del codice e manifest.
- **14 partite complete con replay identico** coprono tutti i dieci profili,
  le quattro condizioni di mercato e il confronto contro avversari forti.
- Archivio ZIP ricostruito dai sorgenti e verificato dopo la creazione.

## Correzioni

1. Domanda elevata: la ricorrenza Poisson non perde più numericamente tutta la
   massa di probabilità, sovrastimando le vendite disponibili.
2. Pianificazione: il beam considera prodotti successivi e capacità condivisa;
   l'allocazione interna dipende dalla capacità effettivamente utilizzabile.
3. Acquisti: riserva monetaria anche per ulteriori unità dello stesso componente.
4. Prezzi: tick decimali piccoli gestiti senza troncamento a dieci cifre.
5. Errori durante una decisione: ripristino del generatore casuale e dei learner.
6. Snapshot: validazione delle referenze incrociate e dei dati delle azioni pendenti.
7. JSON e persistenza: rifiuto di chiavi duplicate, numeri non finiti e stato null;
   manifest di errore e risultati parziali conservati.
8. Confronti: stesso mercato e stessi avversari per i profili; hash dello script
   di studio e copie isolate dei dati passati ai callback di tracing.

Dettagli e limiti residui sono in `docs/AUDIT.md`.

## Valutazione ripetuta con parametri invariati

È stata riutilizzata la configurazione competitive selezionata nella fase di
sviluppo precedente (`demand_prior=3.0`, `demand_optimism=0.3`). Nessuna ricerca
di nuovi parametri è stata fatta sui risultati di questo controllo.

I semi 300000–300019 sono gli stessi dello studio 0.2: si tratta di una verifica
di regressione, **non di un nuovo campione indipendente**. Per ogni profilo:
80 partite contro gli avversari standard e 60 contro EXP3, GP e competitive.
Altre 20 partite confrontano quattro copie del profilo competitive.

| Profilo | Profitto medio, standard | Vittorie | Profitto medio, forti | Vittorie |
|---|---:|---:|---:|---:|
| baseline | 16.66 | 1.2% | 3.49 | 0.0% |
| exp3 | 152.55 | 26.2% | 13.23 | 3.3% |
| gp | 194.93 | 36.2% | 24.99 | 6.7% |
| competitive | 311.64 | 46.2% | 228.94 | 46.7% |

Le percentuali aggregate sono descrittive. Gli stessi semi sono riutilizzati fra
scenari; il CSV riporta medie e dispersione per scenario e confronti appaiati.
Le statistiche descrivono il simulatore locale e non garantiscono il risultato
della competizione ufficiale. I report 0.2 sono archiviati in `reports/legacy-v0.2`.

Il profilo competitive rimane il punto di partenza consigliato; l'adattatore
ufficiale e le ipotesi economiche vanno verificati contro le regole consegnate in gara.
