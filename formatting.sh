#!/bin/sh
black . --extend-exclude .ipynb -v --exclude "/(build|dist|official_baseline_classifier|wav2vec2|official_scoring_metric|helper\_code\.py|run\_model\.py|train\_model\.py|evaluate_model\.py)/"
flake8 . --count --ignore="E501 W503 E203 F841 E402" --show-source --statistics --exclude=./.*,build,dist,wav2vec2,official*,helper_code.py,run_model.py,train_model.py,evaluate_model.py,*.ipynb
