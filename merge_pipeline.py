from training_pipeline.stage1_prepare_data import prepare_data
from training_pipeline.stage2_feature_scale import prepare_data2
from training_pipeline.stage3_data_train import data_trainings
from training_pipeline.stage4_evaluation import evaluation


if __name__ == "__main__":
    prepare_data(config_path="config.yaml")
    prepare_data2(data_yaml="config.yaml")
    data_trainings(data_yaml="config.yaml")
    evaluation(data_yaml="config.yaml")
