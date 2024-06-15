from training_pipeline.stage1_prepare_data import prepare_data
from training_pipeline.stage2_feature_scale import prepare_data2
from training_pipeline.stage3_data_train import data_trainings
from training_pipeline.stage4_evaluation import evaluation
from prediction_pipeline.prediction_pipeline import prediction_for_batch, single_query_prediction


if __name__ == "__main__":
    prepare_data(config_path="config.yaml")
    prepare_data2(data_yaml="config.yaml")
    data_trainings(data_yaml="config.yaml")
    evaluation(data_yaml="config.yaml")
    model_data_path = r'artifacts/models/model.pkl'
    test_tsv_path = r"artifacts/prepared/test.tsv"
    transformer_data_path = r"artifacts/model_features/pipeline.pkl"
    # prediction_for_batch(model_data_path,transformer_data_path, test_tsv_path )
    query = input("Enter Your Query Here:")
    single_query_prediction(model_data_path, transformer_data_path, query)


