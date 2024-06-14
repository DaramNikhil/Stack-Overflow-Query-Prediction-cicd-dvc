from training_pipeline.stage1_prepare_data import prepare_data
from training_pipeline.stage2_feature_scale import prepare_data2


if __name__ == "__main__":
    prepare_data(config_path="config.yaml")
    prepare_data2(data_yaml="config.yaml")
