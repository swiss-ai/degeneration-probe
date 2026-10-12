import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    REPO = Path(__file__).resolve().parents[1]
    OUTPUTS = REPO / "outputs"
    ANALYSIS = OUTPUTS / "analysis"
    FIGURES = Path(__file__).resolve().parent / "figures" / "rebuttal"

    # Dataset builds live on permanent store; scratch only holds run outputs.
    STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
    DATASETS = {
        "apertus-8b-instruct": STORE / "degeneration-dataset-apertus-8b-instruct",
        "llama3p1-8b-instruct": STORE / "degeneration-dataset-llama3p1-8b-instruct",
        "mistral-7b-instruct-v0p1": STORE / "degeneration-dataset-mistral-7b-instruct-v0p1",
    }

    # Training configs of the six S4 probes and the W&B records of their runs.
    RESCUED = STORE / "_rescued_scratch_20261008"
    S4_CONFIGS = RESCUED / "s4_training_configs"
    WANDB_RECORDS = RESCUED / "wandb_configs"
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Rebuttal

    Utilizzerò questo notebook per indicare tutto ciò che desidero venga fatto per migliorare il nostro paper, già accettato a NeurIPS. Prima descriverò lo stato attuale delle cose e poi indicherò, a mio avviso, le azioni necessarie. Successivamente potremo discutere insieme cosa abbia senso fare e cosa no.

    Lo stato attuale è che ho inserito nella cartella *paper* il PDF che abbiamo submitato. Il nostro obiettivo ora è renderlo pronto per essere caricato su Archive, così da poterlo pubblicizzare e conservare una versione finale.

    Stiamo lavorando su Clariden, il supercomputer. Il problema è che gran parte dei dati, probe, training e dataset, sono stati cancellati perché salvati in *scratch*. Di conseguenza i dataset sono stati ricostruiti; al momento dovrebbero essere completi, oppure li stiamo rigenerando e salvando gli stati di attivazione.

    I probe per Apertus sono stati salvati su Aging Face, mentre quelli per Lama e per Mistral devono essere retrenati. L'idea di base era di eseguire tutte le ablation e i vari sweep su Apertus, dove abbiamo già i probe; in caso contrario, potremmo farlo su un unico modello, evitando così il retraining.

    Per ora i dati e gli sweep nel paper saranno mostrati solo su Apertus. Una volta individuata la strategia migliore su Apertus, addestreremo solo la versione con LoRA della strategia migliore. Successivamente, trasferiremo questo training e LoRA anche su Lama e Mistral, richiedendo quindi solo due training aggiuntivi.

    Questi training saranno eseguiti per ogni layer, poiché il layer più efficace può variare a seconda del modello. La validazione dovrà avvenire sull’intero validation split.

    È fondamentale salvare i valori di score per ogni token generato. In questo modo, se dobbiamo calcolare la nuova metrica Token‑Level o Response‑Level, potremo semplicemente prendere il CSV (o il formato più adatto) e ricalcolare la metrica.

    Ad esempio, anche se i probe sono già salvati su Hugging Face, possiamo ri‑eseguire lo scoring sul validation e sul test, ottenere il CSV e ricalcolare tutti i valori. È importante conservare questi score, anche se occupano molto spazio.

    Per il paper dovremo rieseguire tutte le metriche per riconfermarle. Dopo aver completato i training su Lama e Mistral, otterremo i numeri da includere. Oltre alle metriche già presenti (response‑level, AUC, ID e accuracy), vorrei aggiungere:

    - Recall
    - Precision
    - AUROC

    Per le metriche token‑level, desidero includere anche il “Never Fire”, cioè il numero di volte in cui il false negative rate si verifica (una risposta degenere non identificata come tale). Inoltre, a livello token‑level, vorrei mostrare quanti token precedono la warning coverage e vengono segnalati come positivi; questo valore assume maggiore importanza quando la persistent value supera 1.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Diario di bordo

    **10 ottobre 2026**

    - Verificati i tre dataset su store. Prompt (3,630), generazioni (36,300) e label euristiche completi e identici per Apertus, Llama e Mistral. Judge completo a meno dei fallimenti del client cloud: 63 rollout su Apertus, 8 su Llama, 44 su Mistral, esclusi dal pool e non etichettati come negativi.
    - Le onset labels di Mistral erano del 1 settembre, prima della fine del judge: rifatte. Positivi: 1,668 train, 389 val, 318 test in-domain, 132 held-out.
    - Le attivazioni di Llama risultavano presenti nel manifest ma erano state cancellate dal wipe di scratch. Manifest messo da parte e cache rifatta in 2 ore e mezza: 36,300 file, 6.8 TB su capstor scratch. Tutti e tre i dataset hanno ora attivazioni complete.
    - Il wipe aveva cancellato anche l'immagine del container. Ricostruita da un Dockerfile che usa il working tree locale (il repo GitHub è privato e non raggiungibile dai nodi) con le versioni dei pacchetti bloccate a quelle di agosto, perché un pyarrow recente richiede NumPy 2 e l'immagine base ha la 1.26. Script: `cluster/build_local.sh`.
    - Lanciata la sweep frozen su Apertus: 26 ricette x 3 seed = 78 run, 31 teste per run, 3,000 step, nessun early stopping, checkpoint e monitor ogni 100 step su 800 rollout di val (108 positivi + 692 negativi fissi). Output su store. Rimosse le ricette hard-negative, regressione e class balance; aggiunte frontier W=1024 centered K=128/256 e trailing K=128/256/512 (batch 6 per rientrare nel budget di 4,096 token per step).
    - Tempi misurati: 2.3-5.6 s/step a seconda della ricetta, 8 minuti per passaggio di validation. I run all_tokens superano le 8 ore di walltime e vanno ripresi dal checkpoint con `cluster/resume_unfinished.sh`.
    - Scritta la pipeline post-training: replay di tutti i checkpoint sul val completo (`cluster/replay_all.sh`), selezione di uno step per layer (`scripts/select_checkpoints.py`: post-onset >= 0.3 poi massima warning coverage 256; se il layer non raggiunge la soglia, lo step con post-onset massima, marcato non selezionabile), scoring di tutti i 31 layer in un solo passaggio per split con soglie congelate su val (`scripts/score_selected_depths.py`, `cluster/score_selected_launch.sh`). Score per token salvati su store, circa 1.6 GB per run.
    - Smoke test dello scoring su un run reale: 3,634 rollout di val in 494 s per due layer presi da checkpoint diversi, soglie e provenienza corrette.

    **11 ottobre 2026**

    - Esito della sweep: 68 run su 78 finiti a 3,000 step. Nove in timeout a 8 ore (rollout_balanced, random_window W=64, frontier W=64, tre seed ciascuno) e uno fermato da un guasto del nodo (all_tokens K=256, seed 42). Nessun errore di training. I dieci run ripresi dall'ultimo checkpoint (passi 2,000-2,800) con `cluster/resume_unfinished.sh`.
    - La contesa sulle letture da store ha rallentato i run che leggono rollout interi fino a 29 s/step mentre tutti i 78 giravano; con la coda piu' leggera sono tornati a 2-5 s/step.
    - Il replay cercava le attivazioni sotto la root del build invece che nella loro cartella (residuo della separazione di agosto): corretto. Lanciati i replay per i 68 run finiti e, con dipendenza sul training, per i 10 ripresi. Un replay costa circa 75 minuti: legge tutti i 31 layer di ogni rollout di val per intero.
    - Decisione: lo scoring completo con score per token su val, test in-domain e held-out si fa solo sul checkpoint selezionato di ogni layer. Il replay resta solo su val e salva per ogni checkpoint le metriche di selezione e i picchi dei rollout sani, non gli score per token.
    - Commit sul branch dataset-v2-pilot: build del container dal working tree, correzione del replay e caricatore multi-layer, script della sweep, selezione e scoring composito.
    - Replay completati per 75 run su 78 (mancano i tre seed di rollout_balanced, ancora in training). Selezione fatta: 1,983 coppie layer-run selezionabili su 2,325. Il budget di 3,000 step basta: le curve di warning coverage si appiattiscono tra il passo 2,300 e il 2,500.
    - Classifica provvisoria su validation (miglior layer selezionabile per run, media su 3 seed, Warn 256): frontier W=512 centered K=256 0.074, all_tokens K=1024 0.074, all_tokens K=256 0.071, frontier W=512 trailing K=512 0.069; poi le W=1024 e i target soft tra 0.04 e 0.06; in coda frontier W=64 a 0.024. Le prime tre sono dentro una deviazione standard. Il layer 15 vince quasi ovunque in testa. Numeri piu' alti del paper per l'ottimismo di selezione su 30 checkpoint x 31 layer: il confronto vero lo danno i test.
    - Lanciato lo scoring dei 75 run selezionati su val, test in-domain e held-out, 31 layer per run in un passaggio per split, walltime 12 ore per la contesa sulle letture.
    - Partiti su Llama e Mistral i run frozen a 31 layer con la ricetta scelta, frontier W=512 centered K=256, 3 seed per modello. Aggiunti i config modello e i CSV dei rollout di validation dei due build; il replay ora usa il CSV del dataset del run.
    - Partita su Apertus la LoRA per tutti i 31 layer, 3 seed, 93 run: stessa ricetta, rank 16 su q e v di tutti i blocchi fino al layer letto, 3,000 step, validation sul val completo ogni 100 step, nessun early stopping. Serve il profilo frozen vs adapted lungo la profondita' e verifica se l'adattamento sposta il layer migliore. La selezione del checkpoint usa le metriche registrate in training, con la stessa regola e lo stesso fallback.
    - Tutta la catena e' ora in coda con dipendenze: per Llama e Mistral training, replay e scoring; per i LoRA training e scoring adapted (`cluster/score_adapted_selected.sbatch`, che seleziona il checkpoint dalla storia di training e scora il layer del run sui tre split). Un timeout del training blocca il job dipendente, che va rimesso in coda dopo la ripresa.

    - Scoring Apertus completato per 75 run (mancano i 3 rollout_balanced, in coda dietro il loro replay). Letti al layer scelto su validation, budget 1%, Warn 256 medio su 3 seed: frontier W=512 centered K=256 0.074 val / 0.086 test in-domain / 0.167 held-out; all_tokens K=1024 0.073 / 0.095 / 0.159; all_tokens K=256 0.070 / 0.087 / 0.147; frontier W=512 trailing K=512 0.069 / 0.081 / 0.159. Le prime quattro ricette sono le stesse sui tre split, sempre al layer 15; in coda frontier W=64 (0.024 / 0.015 / 0.048). Il test in-domain ha 117 positivi, l'held-out 26: su quest'ultimo i numeri sono indicativi. Tabelle complete in outputs/analysis/selected_metrics.csv e selected_metrics_by_recipe.csv su store.
    - I run LoRA vanno a 9-13 s/step con validation completa in circa 90 s: 10-12 ore per 3,000 step, al limite del walltime massimo della partizione (12 ore). I timeout si riprendono con `cluster/resume_timed_out.sh`, che rimette in coda training e scoring dipendente.

    **12 ottobre 2026**

    - Tutti i training sono finiti senza timeout: 78 frozen Apertus, 6 frozen Llama/Mistral, 93 LoRA Apertus. I LoRA profondi hanno preso 9-10 ore. Replay, selezione e scoring completati per tutto tranne due scoring Mistral (seed 43 e 44), falliti per quota: lo store del progetto infra01 e' al limite di inode (74.96 milioni di file su 75 milioni, condiviso con tutto il gruppo). Rilanciati appena si e' liberato margine.
    - Profilo LoRA per layer su Apertus (frontier W=512 centered K=256, rank 16 sui blocchi fino al layer letto), Warn 256 a budget 1%, media su 3 seed, val / test in-domain: cresce con la profondita', da 0.11 / 0.09 al layer 2 a 0.23 / 0.27 al layer 15, 0.27 / 0.31 al layer 18, 0.29 / 0.31 ai layer 29-30, poi cala al 31 (0.23 / 0.25). Il frozen ha il picco al layer 15 (0.074 / 0.086) e scende a 0.02 oltre il 28. Post-onset adattata tra 0.76 e 0.90 da layer 4 in su. Nel paper l'adapted al layer 15 su test faceva 0.194; ora 0.265, e 0.31 tra il 18 e il 30. Avvertenza per il paper: l'adattatore copre i blocchi 1..L, quindi la capacita' adattata cresce con la profondita' letta.
    - Llama e Mistral frozen stanno sulla soglia: Llama ha 3, 2 e 0 layer selezionabili sui tre seed (post-onset massima 0.30), Mistral 10 sul seed 42 con il migliore al layer 3. Il layer scelto da questi sweep e' rumore intorno a 0.3: per la LoRA su Llama e Mistral serve una scelta esplicita del layer, non la selezione dallo sweep frozen.
    - File su store: 627k miei, di cui 298k nei run frozen Apertus (circa 2,900 file di checkpoint per run, 30 checkpoint x 31 layer). Il replay contiene gia' le metriche di ogni checkpoint; i checkpoint non selezionati sono riaddestrabili.

    **Prossimi passi**

    - Al termine della sweep: resume dei run in timeout, replay, selezione, scoring su val, test in-domain e held-out.
    - LoRA per la ricetta migliore al layer selezionato, 3 seed.
    - Llama e Mistral: config modello, sweep frozen su tutti i layer, poi LoRA.
    - Nuove metriche sui parquet di score: precision, recall, AUROC, never-fire, token segnalati prima della warning band, warning coverage anche a 64 e 32 token (l'evaluator le calcola già insieme a 128 e 256; vanno solo riportate nelle tabelle).
    """)
    return


if __name__ == "__main__":
    app.run()
