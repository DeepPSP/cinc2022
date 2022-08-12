# cinc2022

![docker-ci](https://github.com/wenh06/cinc2022/actions/workflows/docker-image.yml/badge.svg?branch=docker-ci)
![format-check](https://github.com/wenh06/cinc2022/actions/workflows/check-formatting.yml/badge.svg)

Heart Murmur Detection from Phonocardiogram Recordings: The George B. Moody PhysioNet Challenge 2022

## Knowledge about heart murmur
[utils/HeartMurmur.md](/utils/HeartMurmur.md) contains knowledge about heart murmur collected from various sources.

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

## CinC2022 conference paper
~~Folder [tex](/tex) contains latex source code for the CinC2022 conference paper, written using [Overleaf](https://www.overleaf.com/).~~ Moved to another repository as the size of the whole repository exceeds the limit of [Overleaf](https://www.overleaf.com/).
