# Kalbos atpažinimas ir sentimentų analizė

Sistema paverčia kalbą tekstu (Whisper), o tekstą klasifikuoja į teigiamą, neutralų ar neigiamą (Transformer modelis iš Hugging Face). Veikia su lietuvių ir anglų kalbomis, CPU arba GPU.

## Architektūra

```
garsas (failas / mikrofonas)
   -> audio_input.py      tikrina formatą, trukmę, tylą; perkonvertuoja į 16 kHz mono
   -> speech_to_text.py   Whisper: tekstas + nustatyta kalba
   -> sentiment_analysis.py  Transformer klasifikatorius: 3 klasės, tikimybės %
   -> visualization.py    diagrama, įrašas, CSV / JSON failai
main.py = AnalysisPipeline (jungia modulius) + komandinė eilutė
app.py  = Streamlit sąsaja
```

Modeliai įkeliami vieną kartą. Streamlit naudoja `st.cache_resource`, o klasės `SpeechToText` ir `SentimentAnalyzer` laiko įkeltą modelį savo laukuose. Parsisiųsti modeliai saugomi `models/` aplanke.

## Modelių palyginimas

| Modelis | Paskirtis | Privalumai | Trūkumai | Lietuvių kalba |
|---|---|---|---|---|
| Whisper | kalbos atpažinimas | vienas modelis daugeliui kalbų, pats nustato kalbą, atsparus triukšmui | didesni modeliai lėti be GPU | lietuvių kalba palaikoma; kokybė žemesnė nei anglų, todėl geriau rinktis `small` ar `medium` ir patikrinti savo įrašuose |
| BERT (`bert-base-uncased`) | teksto klasifikavimas | gerai ištirtas, daug pavyzdžių | anglų kalbai; be sentimentų galvos reikia pačiam apmokyti | netinka |
| DistilBERT | lengvesnis BERT variantas | mažesnis ir greitesnis už BERT, tinka CPU | pigus sprendimas kainuoja šiek tiek tikslumo; populiarus SST-2 variantas turi tik dvi klases, be neutralios | `distilbert-base-multilingual-cased` egzistuoja, bet sentimentų galvos reikia apmokyti |
| XLM-RoBERTa | daugiakalbis klasifikavimas | išankstinis apmokymas ~100 kalbų tekstais, tarp jų lietuvių | didesnis už DistilBERT | tinkamiausias pagrindas; paruoštas modelis sentimentams nėra specialiai apmokytas lietuvių kalba |

Projekte numatytasis sentimentų modelis yra `cardiffnlp/twitter-xlm-roberta-base-sentiment` (XLM-RoBERTa, klasės negative / neutral / positive). Pagal modelio aprašą jis tikslintas ne lietuvių kalbos duomenimis, todėl lietuviškam tekstui taikomas perkėlimas iš kitų kalbų (zero-shot). Prieš pasitikint rezultatais, išbandykite jį su savo pavyzdžiais. Anglų kalbai galima pasirinkti `english` modelį (`cardiffnlp/twitter-roberta-base-sentiment-latest`).

### Jei lietuvių kalbos tikslumo nepakanka

1. Surinkite lietuviškų sakinių su ranka priskirtomis klasėmis (positive / neutral / negative).
2. Padalykite duomenis į apmokymo ir testavimo dalis (pvz. 80/20).
3. Tikslinkite `xlm-roberta-base` su Hugging Face `Trainer`.
4. Palyginkite F1 ir tikslumą su paruoštu modeliu tame pačiame testavimo rinkinyje.
5. Naują modelį įrašykite į `config.SENTIMENT_MODELS`.

## Diegimas

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Rekomenduojama įdiegti `ffmpeg` (formatams kaip M4A ar MP3 dekoduoti). GPU naudojimui įdiekite PyTorch versiją su CUDA pagal pytorch.org instrukcijas; kodas GPU pasirenka automatiškai.

## Paleidimas

Sąsaja:

```bash
streamlit run app.py
```

Komandinė eilutė:

```bash
python main.py garsas.wav --language lt --whisper-model small
```

Pirmą kartą paleidus modeliai parsisiunčiami, tam reikia interneto ir vietos diske.

## Testai

```bash
python -m pytest tests
```

Testai tikrina garso įkėlimą, validaciją, etikečių normalizavimą ir rezultatų eksportą. Jiems nereikia parsisiųsti modelių.

## Struktūra

```
speech_sentiment_analysis/
  app.py  main.py  config.py  requirements.txt  README.md
  modules/  audio_input.py  speech_to_text.py  sentiment_analysis.py  visualization.py
  models/   (parsisiųsti modeliai)
  results/  (JSON, CSV, PNG)
  tests/    test_core.py
```

## Apribojimai

- Įrašo trukmės riba 600 s (keičiama `config.py`).
- Sentimentų modelis nuskaito iki 512 žetonų; ilgesnis tekstas sutrumpinamas.
- Whisper kalbos nustatymas automatiniu režimu gali suklysti trumpuose įrašuose; tokiu atveju pasirinkite kalbą ranka.

## Plėtra

- Emocijų klasifikavimas: pridėti modelį su klasėmis anger, joy, sadness, fear ir nauja klasė `EmotionAnalyzer` šalia `SentimentAnalyzer`.
- Atskirų sakinių analizė: Whisper jau grąžina `segments`; juos galima siųsti į `SentimentAnalyzer` po vieną.
- Istorija: `save_results` jau rašo kiekvieną analizę į `results/`; liko sujungti failus į vieną lentelę.
- Tikslumo vertinimas: `scikit-learn` `classification_report` su anotuotu testavimo rinkiniu.
