from src.components.inference import Model
from src.components.library.vectors import VectorDb
from src.config.configuration import ConfigurationManager
# model=Model()
# model.predict(image_path)

config=ConfigurationManager()
lib_config=config.get_library_components()
vec=VectorDb(lib_config)

vec.searchQuery("D:\\Ml Dl\\Project\\LunaCraterNet\\artifacts\\crops\\-100.43091128943604,-99.05903044959871,-11.519182147960857,-10.147301308123556_crop_26.jpg")

