from src.components.inference import Model
from src.components.model_trainer_2 import Cropper
from src.components.library.ResNet import Embedding
from src.components.library.vectors import VectorDb
from src.config.configuration import ConfigurationManager
image_path="D:\\Ml Dl\\Project\\LunaCraterNet\\artifacts\\data_ingestion\\dataset\\LU3M6TGT_yolo_format\\train\\images\\-0.28360998131250864,1.0882708585247933,-11.519182147960857,-10.147301308123556.png"

# model=Model()
# model.predict(image_path)

# im=Cropper(image_path)
# im.crop()
# config=ConfigurationManager()
# lib_config=config.get_library_components()
# emb=Embedding(lib_config)
# a,b=emb.run()
# print(a)

vec=VectorDb()
# vec.storeDb()
vec.searchQuery("D:\\Ml Dl\\Project\\LunaCraterNet\\artifacts\\crops\\crop_13.jpg")

