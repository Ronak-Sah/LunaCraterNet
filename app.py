from src.components.inference import Model

image_path="D:\\Ml Dl\\Project\\LunaCraterNet\\artifacts\\data_ingestion\\dataset\\LU3M6TGT_yolo_format\\train\\images\\-0.28360998131250864,1.0882708585247933,-11.519182147960857,-10.147301308123556.png"

model=Model()
model.predict(image_path)