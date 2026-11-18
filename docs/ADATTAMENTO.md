# Collegamento al gioco ufficiale

## Scheda da compilare al briefing

| Informazione | Da registrare |
|---|---|
| Punteggio | Profitto, liquidità, patrimonio, vendite; trattamento dell'invenduto |
| Tempo | Turni totali, ordine delle fasi, latenza del feedback, timeout per chiamata |
| Acquisti | Primo/secondo prezzo, riserva, pareggi, unità, lotti, pagamento e rimborso |
| Produzione | Ricette, capacità, costi, tempi, scorte iniziali, deperibilità |
| Prezzi | Estremi, tick, valuta/unità, possibilità di non vendere |
| Informazioni | Domanda, visite, vendite, costi, prezzi/offerte avversarie osservabili |
| Budget | Fisso o rinnovabile, credito, liquidità impegnata, costi futuri obbligatori |
| Esecuzione | Callback o stdin/stdout, persistenza, librerie, rete, memoria, CPU |
| Regole gara | Codice preparato consentito, test disponibili, formato e scadenza consegna |

## Sequenza operativa

1. Salvare una configurazione distinta con le regole confermate. Non trattare i
   default del simulatore come specifica della gara.
2. Implementare un adattatore seguendo `MarketAdapter`. Tradurre unità e nomi,
   richiedere esplicitamente i dati necessari, aggregare feedback parziali e
   assegnare identificativi univoci a tutte le decisioni.
3. Se il server mantiene il processo, istanziare `MarketAgent` una volta e chiamare
   `act`/`observe`. Se riavvia il processo, ripristinare `state_dict` fra invocazioni.
   Lo stato JSON è versionato e richiede la stessa configurazione.
4. Adeguare `validate_action` ai vincoli ufficiali. Se le fasi hanno un ordine
   diverso, modificare il coordinamento e i dati disponibili prima delle decisioni.
   Cambiare solo la serializzazione non risolve una diversa dinamica del gioco.
5. Acquisire una traccia reale minima e verificare la baseline su una partita intera.
   Controllare ogni pagamento, trasferimento di scorte e componente del punteggio.
6. Abilitare i learner compatibili; scegliere griglia e finestra in base ai turni
   effettivi e alle opportunità di feedback, non alla durata in ore dell'hackathon.
7. Eseguire prove consentite con semi/scenari separati dalla selezione dei parametri.
   Conservare baseline, configurazione scelta e comando esatto di consegna.

## Protocollo JSON interno

`act` legge un oggetto con `observation` e lista facoltativa `feedback`. I feedback
si applicano prima della nuova decisione. Su stdout esce un solo oggetto `Action`;
lo stato viene salvato nel percorso `--state` e gli errori vanno su stderr con exit 2.

```json
{
  "feedback": [{
    "decision_id": "example:auction:0",
    "phase": "auction",
    "won": true,
    "costs": {"metal": 2.0}
  }],
  "observation": {
    "decision_id": "example:production:0",
    "turn": 0,
    "phase": "production",
    "cash": 148.0,
    "raw": {"metal": 1, "plastic": 0},
    "stock": {"widget": 0, "gadget": 0},
    "raw_cost": {"metal": 2.0, "plastic": 1.0},
    "stock_cost": {"widget": 0.0, "gadget": 0.0},
    "capacity": 5
  }
}
```

Per pricing, `rewards` contiene il margine realizzato per prodotto. Per conversion
TS servono inoltre `censored` e, quando non censurato, `successes` e `trials`.
Il profilo `competitive` richiede `sales` e `censored` per ogni prodotto: la
scorta offerta e il prezzo corrispondenti sono già conservati nella decisione pendente.
Se il feedback ufficiale contiene soltanto il risultato totale, serve cambiare il
learner oppure definire una decomposizione giustificata; non inventare contributi
per prodotto. Se i costi non sono forniti, mantenerne una stima esplicita nel proprio
adattatore, distinguendola da una misura ufficiale.

I prezzi dei learner sono fissi per episodio. Un cambiamento dei limiti osservati
che escluda un prezzo configurato provoca un errore: ricostruire la griglia e
resettare oppure implementare un trasferimento esplicito delle statistiche.

## Punti di attenzione concreti

- Budget pacing: il ritmo iniziale usa turni × aste nominali per turno. Se le
  opportunità effettive variano, adattare il calcolo al calendario reale.
- Produzione e offerte: i profili semplici usano prezzi precedenti e costi stimati;
  `competitive` usa già un modello di domanda approssimato. Verificarne le ipotesi
  e la scala, senza assumere che la domanda ufficiale segua lo stesso modello.
- Feedback ritardato: prenotare la liquidità per ordini pendenti se il server non
  restituisce già il saldo spendibile. Il simulatore risolve ogni asta subito.
- Scegliere limiti di ricompensa che includano perdite, costi e quantità possibili.
- Le cache di azioni/decisioni crescono con l'episodio. In una gara molto lunga,
  definire col server una finestra di idempotenza e una politica di compattazione.
- Il simulatore locale consente tuning e debug; la sua classifica non dimostra
  quale strategia vincerà nell'ambiente ufficiale.
