# Audit 0.2.1 e preparazione competitiva

## Ricontrollo del 18 novembre 2026

Sono stati identificati e corretti ulteriori difetti non intercettati dalla suite iniziale:

- La massa di probabilità Poisson partiva da `exp(-lambda)`: con lambda elevato
  andava numericamente a zero e il modello sovrastimava le vendite anche con scorte
  superiori alla domanda media. La ricorrenza ora usa log-probabilità e limiti coerenti.
- Una ricerca con beam molto stretto poteva consumare capacità sul primo prodotto
  e scartare un prodotto successivo molto più remunerativo. Ora considera un limite
  superiore sul contributo dei prodotti rimanenti e segnala se ha potato stati.
- Il budget di una singola asta riservava gli altri componenti, ma non le unità
  ulteriori dello stesso componente. Anche queste sono ora riservate.
- Un errore durante la seconda scelta di prezzo poteva lasciare consumati il
  generatore casuale e la prima scelta. Le decisioni fallite ora vengono annullate.
- Uno snapshot poteva avere decisioni pendenti incoerenti fra agente e learner,
  perdendo silenziosamente un aggiornamento. Le referenze incrociate sono validate.
- Arrotondamenti fissati a dieci decimali e tolleranze assolute alteravano tick
  piccoli. I controlli condividono ora un'aritmetica decimale esplicita.
- JSON con chiavi ripetute o numeri non finiti poteva essere accettato; uno stato
  `null` poteva azzerare l'agente. Questi input ora producono errori espliciti.
- Nel protocollo di studio una selezione con mercato modificato poteva essere
  confrontata con baseline sul mercato predefinito. Ora mercato e avversari sono
  identici e registrati per tutti i profili del confronto.
- Un callback di logging poteva modificare riferimenti al mercato o ai risultati:
  le tracce ora ricevono copie isolate. Errori di persistenza vengono registrati
  nel manifest e lo studio conserva i risultati parziali.

I parametri del profilo competitive sono rimasti quelli già selezionati. La
valutazione è stata ripetuta per verificare la nuova build, senza ritoccare i
parametri in base ai risultati. I nuovi test e i conteggi sono in [VERIFICA.md](../reports/VERIFICA.md).

L'audit ha esaminato contratti, learner, agente, contabilità, simulatore e metodo
di valutazione. Le correzioni sono accompagnate da regressioni automatiche;
questo non costituisce una prova dell'assenza di qualsiasi difetto possibile.

## Problemi corretti

| Problema verificato | Correzione |
|---|---|
| UCB iniziava sempre dal prezzo più basso | Esplorazione e pareggi randomizzati, con generatore salvato |
| Feedback ritardato concentrava esplorazione sullo stesso prezzo | Conteggio delle decisioni pendenti |
| GP assumeva distanze uguali fra prezzi anche dopo arrotondamento | Kernel sulle coordinate reali, risoluzione numerica con jitter |
| Scansione ripetuta di tutta la storia per le statistiche dei learner | Statistiche sufficienti aggiornate incrementalmente |
| Feedback duplicato poteva aggiornare direttamente il learner due volte | Consumo idempotente della decisione, anche dopo censura/scadenza |
| Caricamento di stato parziale o con alias mutabili | Ripristino validato in staging, copie isolate e schema esplicito |
| Identificativo riutilizzato con osservazione diversa | Impronta dell'osservazione; conflitto rifiutato |
| Duplicato contraddittorio accettato | Impronta del feedback; conflitto rifiutato |
| Dati NaN, bool numerici, limiti e pagamenti incoerenti | Controlli espliciti sui contratti e sui vincoli economici |
| Limiti dei prezzi mancanti diventavano sempre 5/30 | Valori risolti dalla configurazione, non da costanti nel messaggio |
| Profili parziali potevano mescolare strategie incompatibili | Selezione completa dei tre moduli |
| Tuning sovrascriveva il profilo configurato | Profilo conservato, griglie generiche e semi separati |
| Benchmark senza identità del codice né confronti appaiati | Hash dei sorgenti, manifest con stato, differenze per seme |
| Risultati dipendevano dall'ordine delle chiavi dei dizionari | Ordine canonico nel simulatore |
| Valutazione solo contro avversari semplici | Roster espliciti, rank, vittorie e prove fra copie dell'agente |

## Nuova strategia coordinata

Il profilo `competitive` aggiunge un modello di domanda con censura: vendere tutte
le unità disponibili informa che la domanda era almeno quella quantità. La
verosimiglianza usa code Poisson calcolate stabilmente anche quando molto rare.
I prezzi massimizzano ricavi attesi limitati dalle scorte. Per nuovi prodotti, il
piano confronta ricavi incrementali e costi di materie prime e produzione.

Gli acquisti finanziano un piano comune sotto capacità e budget, considerando
il costo dei componenti complementari e riservando denaro per produrre. Si evita
così di trattare ogni materia prima come investimento indipendente. Nelle aste
al primo prezzo, le offerte sono adattate alle stime osservate dei costi; la
formula al secondo prezzo resta distinta.

## Limiti ancora rilevanti

- La domanda Poisson log-lineare è un modello approssimato. Non vi sono garanzie
  di regret o di vittoria per il gioco intero; ipotesi e scala vanno riviste in gara.
- Il simulatore non riassegna domanda insoddisfatta ad altri venditori, valuta
  l'invenduto al costo e limita le spese di magazzino alla liquidità. Queste regole
  sono esplicite; sostituirle se il regolamento ufficiale è diverso.
- La pianificazione è miope su un turno, con scorte obiettivo. Non risolve un
  problema dinamico completo né negozia contro offerte future note.
- Il beam limita il costo della ricerca per molti prodotti: può perdere l'ottimo.
  Il modello di domanda va dimensionato prima di usare inventari di scala molto
  maggiore dei casi testati.
- Le cache idempotenti crescono con l'episodio; prima di episodi lunghi va definita
  la finestra ufficiale entro cui possono arrivare duplicati o feedback ritardati.
- La liquidità osservata deve essere spendibile: eventuali ordini pendenti e
  feedback parziali del server vanno gestiti nell'adattatore.
- Un aumento del profitto medio non basta per scegliere: il report include
  vittorie, dispersione, ranghi, liquidità e differenze appaiate.

## Compatibilità e riproduzione

Snapshot agente: schema 2. Gli snapshot 0.1 non sono migrati implicitamente.
Per riprodurre tracce precedenti serve il vecchio archivio, non incluso nel repository; non confrontare
numericamente benchmark eseguiti con due diverse versioni del simulatore.
La configurazione selezionata è congelata prima della valutazione su semi nuovi.
Correzioni puramente numeriche e di idempotenza sono state riverificate sugli
stessi semi senza riottimizzare i parametri.
