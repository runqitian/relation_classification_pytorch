from cnn_model import KejModel
from data_tools import DataTools
from preprocess import  PreTrain

kjml = KejModel()
# kjml.train_model(True,"model/kej_model.h5",load = True)
# kjml.model_predict("pkl/test.pkl.gz")
kjml.load_kej_model("model/kej_model.h5")