from src.logger import logger 
from multiprocessing import freeze_support

from src.pipeline.stage_01_data_ingestion import Data_Ingestion_pipeline
from src.pipeline.stage_02_model_trainer import Model_Trainer_pipeline
from src.pipeline.stage_03_model_evaluation import Model_Evaluation_pipeline
from src.pipeline.stage_04_storevectors import vector_db_pipeline

def main():
    logger.info("Code Starts")
    # data_ingestion=Data_Ingestion_pipeline()
    # data_ingestion.main()
    model_trainer=Model_Trainer_pipeline()
    model_trainer.main()
    model_eval=Model_Evaluation_pipeline()
    model_eval.main()
    vector_db=vector_db_pipeline()
    vector_db.main()



    
    

if __name__ == "__main__":
    freeze_support()   
    main()
