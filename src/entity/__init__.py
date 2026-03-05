from dataclasses import dataclass
from pathlib import Path

# Configuration for data ingestion

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_url :str
    local_data_file :Path
    unzip_dir :Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir:Path
    data_path :Path
    model_path :Path
    epochs : int
    batch_size :int
    

@dataclass(frozen=True)
class ModelEvalConfig:
    root_dir:Path
    data_path :Path
    model_path : Path
    batch_size :int
    
@dataclass(frozen=True)
class LibraryConfig:
    root_dir:Path
    data_path :Path
    filenames_path : Path
    index_path :Path
    crop_save_dir :Path
    