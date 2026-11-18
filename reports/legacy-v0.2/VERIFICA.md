# Verifica della versione 0.2

## Correzioni e controlli

- **157 test passati** nell'ambiente completo.
- **155 passati, 2 saltati** nel virtualenv minimo: i due controlli opzionali
  richiedono SciPy (solver MILP e confronto numerico della coda Poisson).
- Installazione del pacchetto 0.2 verificata in un virtualenv Windows separato;
  CLI, simulazione con avversari forti e replay eseguiti fuori dalla cartella sorgente.
- **10 tracce di partite complete** della build finale ricostruite con azioni e
  stato identici; inoltre CLI/checkpoint sono verificati nei test.
- Nessuna azione invalida nelle **112 partite di sviluppo** e nelle **580 partite
  di valutazione**; identità contabile ricontrollata in tutte le 580 righe finali.

L'audit dettagliato e i limiti residui sono in `docs/AUDIT.md`.

## Selezione e valutazione

Quattro configurazioni del modello sono state confrontate sui semi 200–203,
con quattro scenari contro avversari standard e tre scenari contro avversari
forti: 112 partite. Criterio fissato: massima quota di vittorie, poi profitto medio.
È stato selezionato `demand_prior=3.0`, `demand_optimism=0.3`; gli altri parametri
sono quelli dichiarati in `configs/competitive.toml`.

La configurazione è stata congelata e confrontata su **20 semi nuovi,
300000–300019**. Per ciascun profilo: 80 partite contro gli avversari standard,
60 contro un gruppo composto da EXP3, GP-UCB e una copia di competitive.
Ulteriori 20 partite confrontano quattro copie identiche di competitive.
Lo scenario aggressivo è escluso dal gruppo forte, perché non cambia quel roster.

Le correzioni finali di precisione numerica/idempotenza sono state verificate
ripetendo la valutazione sugli stessi semi, senza scegliere nuovi parametri.
Il report contiene esclusivamente i risultati della build finale, il cui hash
coincide con il manifest di valutazione. La ripetizione non viene conteggiata
come evidenza indipendente aggiuntiva.

## Risultati

| Profilo | Profitto, avversari standard | Vittorie | Profitto, avversari forti | Vittorie |
|---|---:|---:|---:|---:|
| baseline | 16.66 | 1.2% | 3.49 | 0.0% |
| exp3 | 152.55 | 26.2% | 13.23 | 3.3% |
| gp | 194.93 | 36.2% | 24.99 | 6.7% |
| competitive | 311.64 | 46.2% | 228.94 | 46.7% |

Nel confronto fra quattro copie identiche, l'agente nella posizione osservata
vince il 20.0% delle 20 partite.
È un controllo del funzionamento, non una stima precisa di un vantaggio di posizione.

Le percentuali aggregate sono descrittive: gli stessi 20 semi vengono riusati sui
diversi scenari. `holdout-summary.csv` separa i casi; `holdout-paired.csv` riporta
le differenze di profitto rispetto agli altri profili sugli stessi semi, con errore
standard per scenario. Non si deduce significatività dalla sola media aggregata.

Il guadagno riguarda **questo simulatore**, con le sue ipotesi esplicite su domanda,
scorte e aste. Non garantisce vittorie nell'hackathon. Il profilo competitive è
la base raccomandata per le prove; conservare EXP3 e GP come alternative e
riconfrontarli appena saranno noti regolamento e ambiente ufficiali.

I vecchi risultati 0.1 sono archiviati in `reports/legacy-v0.1` e non sono
confrontabili direttamente: è cambiato anche l'ordine canonico del simulatore.
