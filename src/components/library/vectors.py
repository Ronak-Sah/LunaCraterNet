from src.components.library.ResNet import Embedding
import faiss
from src.config.configuration import ConfigurationManager
import numpy as np
import json
import os
from pathlib import Path
from src.components.model_trainer_2 import Cropper
class VectorDb:
    def __init__(self):
        config=ConfigurationManager()
        self.lib_config=config.get_library_components()
        

    def storeDb(self):
        emb=Embedding(self.lib_config)
        vectors,filenames=emb.run()
        vectors = vectors.astype(np.float32) 

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)

        index.add(vectors)


        faiss.write_index(index, self.lib_config.index_path)
        print(f"Index saved to: {self.lib_config.index_path}")

        with open(self.lib_config.filenames_path, "w") as f:
            json.dump(filenames, f)
        print(f"Filenames saved to: {self.lib_config.filenames_path}")


    def searchQuery(self,image_path:Path):

        cp=Cropper(image_path)
        image=cp.crop()

        emb=Embedding(self.lib_config)
        vectors=emb.embed_single(image)
        print("svec",vectors.shape)
        svec = vectors.astype(np.float32) 
        print("svec",svec.shape)
        index = faiss.read_index("artifacts\\library\\vectorDB")


        distance,pos = index.search(svec,k=2)

        with open(self.lib_config.filenames_path, "r") as f:
            filenames = json.load(f)

        results = [filenames[i] for i in pos[0]]
        print(results)
        return results


