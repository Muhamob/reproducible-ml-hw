.PHONY: all add_feature_extraction add_feature_extraction add_eval_model add_create_model
TRAIN_PATH=data/raw/train.csv.gz
PARAMS_PATH=src/params.yaml

all: clean add_feature_extraction add_create_model add_eval_model add_fit_model

clean:
	rm -f dvc.*

add_feature_extraction:
	dvc run --force -n feature_extraction \
		-d src/load_data.py \
		-d src/feature_extraction.py \
		-d $(PARAMS_PATH) \
		-d $(TRAIN_PATH) \
		-o models/feature_extractor.pickle \
		-o data/processed/features.csv \
		PYTHONPATH=. python src/feature_extraction.py --params $(PARAMS_PATH)

add_create_model:
	dvc run --force -n create_model \
		-d src/create_model.py \
		-d $(PARAMS_PATH) \
		-o models/model_base.pickle \
		PYTHONPATH=. python src/create_model.py --params $(PARAMS_PATH)

add_eval_model:
	dvc run --force -n eval_model \
		-d src/evaluate_model.py \
		-d $(PARAMS_PATH) \
		-d models/feature_extractor.pickle \
		-d models/model_base.pickle \
		PYTHONPATH=. python src/evaluate_model.py --params $(PARAMS_PATH)

add_fit_model:
	dvc run --force -n fit_model \
		-d src/fit_model.py \
		-d $(PARAMS_PATH) \
		-d models/model_base.pickle \
		-d data/processed/features.csv \
		-o models/model.pickle \
		PYTHONPATH=. python src/fit_model.py --params $(PARAMS_PATH)

plot_dag:
	dvc dag --dot | dot -Tpng  -o dag.png