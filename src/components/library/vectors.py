from src.components.library.ResNet import Embedding
import faiss
from src.config.configuration import ConfigurationManager
import numpy as np
import json
from src.logger import logger
from pathlib import Path
from src.components.library.Resize import Cropper
from src.entity import LibraryConfig
class VectorDb:
    def __init__(self,config:LibraryConfig):
        self.lib_config=config
        

    def storeDb(self):
        emb=Embedding(self.lib_config)
        vectors,filenames=emb.run()
        vectors = vectors.astype(np.float32) 

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)

        index.add(vectors)


        faiss.write_index(index, self.lib_config.index_path)
        logger.info(f"Index saved to: {self.lib_config.index_path}")

        with open(self.lib_config.filenames_path, "w") as f:
            json.dump(filenames, f)
        logger.info(f"Filenames saved to: {self.lib_config.filenames_path}")


    def searchQuery(self,image_path:Path):

        cp=Cropper(image_path,self.lib_config)
        image=cp.crop()
        if image is None:
            print("No crater detected in query image")
            return
        emb=Embedding(self.lib_config)
        vectors=emb.embed_single(image)

        svec = vectors.astype(np.float32) 

        index = faiss.read_index(self.lib_config.index_path)


        _,pos = index.search(svec,k=2)

        with open(self.lib_config.filenames_path, "r") as f:
            filenames = json.load(f)

        results = [filenames[i] for i in pos[0]]
        print(results)
        return results


