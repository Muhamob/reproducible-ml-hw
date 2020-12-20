.PHONY: all add_feature_extraction add_feature_extraction add_eval_model add_create_model

all: add_create_model add_eval_model add_feature_extraction add_fit_model

add_feature_extraction:
	dvc run -n feature_extraction \
		-d src/load_data.py \
		-d src/feature_extraction.py \
		-d src/params.yaml \
		-d data/raw/train.csv \
		-o models/feature_extractor.pickle \
		-o data/processed/features.csv \
		PYTHONPATH=. python src/feature_extraction.py

add_create_model:
	dvc run -n create_model \
		-d src/create_model.py \
		-d src/params.yaml \
		-o models/model_base.pickle \
		PYTHONPATH=. python src/create_model.py

add_eval_model:
	dvc run -n eval_model \
		-d src/evaluate_model.py \
		-d src/params.yaml \
		-d models/model_base.pickle \
		PYTHONPATH=. python src/evaluate_model.py

add_fit_model:
	dvc run -n fit_model \
		-d src/fit_model.py \
		-d src/params.yaml \
		-d models/model_base.pickle \
		-o models/model.pickle \
		PYTHONPATH=. python src/fit_model.py
