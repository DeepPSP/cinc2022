# cinc2022

![docker-ci](https://github.com/wenh06/cinc2022/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)
![format-check](https://github.com/wenh06/cinc2022/actions/workflows/check-formatting.yml/badge.svg)

Heart Murmur Detection from Phonocardiogram Recordings: The George B. Moody PhysioNet Challenge 2022

## Knowledge about heart murmur

[utils/HeartMurmur.md](/utils/HeartMurmur.md) contains knowledge about heart murmur collected from various sources.

## Top team papers

**to add after conference....**

## Official phase leaderboards

[Murmur](https://docs.google.com/spreadsheets/u/0/d/e/2PACX-1vRNBATogMRsfio3938bU4r6fcAad85jNzTbSRtRhQ74xHw9shuYoP4uxkK6uKV1zw8CKjPC3AMm33qn/pubhtml/sheet?headers=false&gid=0)
[Outcome](https://docs.google.com/spreadsheets/u/0/d/e/2PACX-1vRNBATogMRsfio3938bU4r6fcAad85jNzTbSRtRhQ74xHw9shuYoP4uxkK6uKV1zw8CKjPC3AMm33qn/pubhtml/sheet?headers=false&gid=1883863848)

The leaderboards can be loaded via

```python
# beautifulsoup4 and html5lib required
import pandas as pd

outcome_url = "https://docs.google.com/spreadsheets/u/0/d/e/2PACX-1vRNBATogMRsfio3938bU4r6fcAad85jNzTbSRtRhQ74xHw9shuYoP4uxkK6uKV1zw8CKjPC3AMm33qn/pubhtml/sheet?headers=false&gid=1883863848"
murmur_url = "https://docs.google.com/spreadsheets/u/0/d/e/2PACX-1vRNBATogMRsfio3938bU4r6fcAad85jNzTbSRtRhQ74xHw9shuYoP4uxkK6uKV1zw8CKjPC3AMm33qn/pubhtml/sheet?headers=false&gid=0"

df_outcome = pd.read_html(outcome_url, flavor="bs4", header=[1], index_col=[0])[0].reset_index(drop=True).dropna()
df_outcome.Rank = df_outcome.Rank.astype(int)
# df_outcome.set_index("Rank", inplace=True)  # Rank has duplicates
df_murmur = pd.read_html(murmur_url, flavor="bs4", header=[1], index_col=[0])[0].reset_index(drop=True).dropna()
df_murmur.Rank = df_murmur.Rank.astype(int)
# df_murmur.set_index("Rank", inplace=True)  # Rank has duplicates
```

pattern for the content of email announcing the submission scores:

```python
from string import punctuation

team_name_pattern = f"""[\\w\\s{punctuation}]+"""
email_pattern = (
    f"""We processed an entry from Team (?P<team_name>{team_name_pattern}) """
    """for the Official phase of the George B\\. Moody PhysioNet Challenge 2022\\. """
    """This entry was submitted on (?P<submission_time>[\\d]{1,2}/[\\d]{1,2}/2022 [\\d]{1,2}:[\\d]{1,2}:[\\d]{1,2} ET) """
    f"""with ID (?P<submission_id>{team_name_pattern}_[\\d]{{1,5}}_[\\d]{{1,3}})\\.[\\n]+"""
    """We successfully evaluated your entry, which received the score (?P<outcome_cost>[\\d\\.]+) and """
    """(?P<murmur_weighted_accuracy>[\\d\\.]+) using the Challenge evaluation metric on the validation set\\. """
    """This entry was your team's (?P<submission_number>[\\d]{1,2})/10 entry for the Official phase\\."""
)
# usage:
# list(re.finditer(email_pattern, email_content))[0].groupdict()
```

## Python re-implementation of Springer's PCG features extractor

[`pcg_springer_features`](https://github.com/DeepPSP/pcg_springer_features) re-implements the feature extraction part of [David Springer's logistic regression-HSMM-based reart sound segmentation algorithm](https://physionet.org/content/hss/1.0/).

Inside [utils](/utils) there's also a copy of `pcg_springer_features`.

## CinC2022 conference paper

~~Folder [tex](/tex) contains latex source code for the CinC2022 conference paper, written using [Overleaf](https://www.overleaf.com/).~~ Moved to another repository as the size of the whole repository exceeds the limit of [Overleaf](https://www.overleaf.com/).

**to add URLs after conference....**

## Poster

**to add....**
