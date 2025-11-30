from src.ml.model import SupplierRiskModel


def get_model(path_to_model):
    ml_model = SupplierRiskModel()
    ml_model.load_model(filepath=path_to_model)
    return ml_model