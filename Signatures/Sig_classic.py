import numpy as np
import signature.path_signature as ps

class SignatureClassique:
    def __init__(self, path, trunc):
        self.path = path
        self.trunc = trunc
        self.signature = None

    def calculer(self):
        
        self.signature = ps.path_to_signature(self.path, self.trunc)
        return self.signature

