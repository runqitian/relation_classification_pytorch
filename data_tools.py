import gzip
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

class DataTools(object):
    def read_pklzip(self,filename):
        print(filename)
        f = gzip.open(filename,"rb")
        data = pkl.load(f)
        f.close()
        return data

    def explain_data(self,data,pflag = False):
        ys = data['relationidxs']
        tokenMatrix = data['tokenMatrix']
        positionMatrix1 = data['positionMatrix1']
        positionMatrix2 = data['positionMatrix2']
        sdpMatrix = data['sdpMatrix']
        if pflag:
            print("tokenMatrix: ", tokenMatrix.shape)
            print("positionMatrix1: ", positionMatrix1.shape)
            print("positionMatrix2: ", positionMatrix2.shape)
            print("ys: ", ys.shape)
            print("sdpMatrix: ", sdpMatrix.shape)
        return ys,tokenMatrix,positionMatrix1,positionMatrix2,sdpMatrix

    def get_data(self,filepath):
        data = self.read_pklzip(filepath)
        return self.explain_data(data,False)