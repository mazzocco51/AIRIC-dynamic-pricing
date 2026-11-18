# Verifica della consegna

Verifica eseguita su Windows con Python 3.12.8.

- **57 test passati** nell'ambiente completo, inclusa l'ottimizzazione SciPy.
- **56 test passati, 1 saltato** nel virtualenv pulito con NumPy e pytest; il test
  saltato richiede l'extra opzionale SciPy.
- Pacchetto installato nel virtualenv pulito; CLI eseguita da fuori dalla cartella
  sorgente, con import verificato da `site-packages`. Simulazione e replay riusciti.
- **680 partite complete**, tutte senza azioni invalide e con identità contabile verificata.
- **21 replay di tracce complete** verificati, con azioni e stato finale identici.
- Benchmark principale: 640 partite, quattro profili e quattro scenari, 20 semi
  di sviluppo e 20 distinti di valutazione per combinazione.
- Verifiche aggiuntive: 24 partite GP/conversione/MILP, 8 aste al primo prezzo e
  8 pacing su budget fisso. Questi campioni ridotti verificano il funzionamento,
  non supportano confronti statistici con il benchmark principale.
- CLI di tuning verificata con una griglia minima nei test. Non è stato eseguito
  un tuning esteso per scegliere la strategia da usare in gara.

## Profitto sui semi di valutazione

Media ± errore standard su 20 partite per cella; unità monetarie del simulatore.

| Profilo | Stabile | Shock | Scarsità | Aggressivo |
|---|---:|---:|---:|---:|
| baseline | 35.72 ± 25.13 | 49.74 ± 25.87 | 33.29 ± 16.10 | 1.68 ± 0.19 |
| adaptive | 72.69 ± 51.35 | 63.98 ± 38.99 | 98.02 ± 54.43 | -1.16 ± 0.17 |
| ucb | 74.49 ± 53.07 | 61.93 ± 37.06 | 87.87 ± 49.28 | -1.16 ± 0.17 |
| exp3 | 225.36 ± 73.27 | 170.63 ± 48.33 | 272.45 ± 65.26 | 30.34 ± 10.46 |

Questi risultati dipendono dalle regole didattiche, inclusa la valutazione delle
scorte al costo. Il pricing è condizionato dalla disponibilità delle materie prime:
controllare anche vendite, scorte e liquidità. La presenza di learner avanzati non
garantisce un miglioramento della strategia completa. Il profilo predefinito resta
`baseline`; la scelta per la gara richiede prove sull'ambiente ufficiale.

Tempi decisionali, deviazioni standard e decimi percentili sono nei CSV allegati.
Le misure temporali non sono riproducibili esattamente e non rappresentano un
limite garantito sul computer della gara. Configurazioni, versioni, semi e tracce
complete si trovano in `runs/`; i dettagli della verifica in `verification.json`.
