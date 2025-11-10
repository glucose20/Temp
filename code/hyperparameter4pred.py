from datetime import datetime


class HyperParameter:
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')                      
        self.word2vec_pth = './data/model_300dim.pkl'
        
        # predict file info
        self.pred_dataset = 'simple-case'
        self.sep = ','
        
        self.pred_pair_pth = './data/simple-Case/predict.csv'
        self.pair_col_name = ['drug_id', 'prot_id','drug_smile', 'prot_seq']
        
        self.pred_drug_dir = './data/EGFR-Case/drug.tsv'
        self.pred_prot_dir = './data/EGFR-Case/prot.tsv'
        self.d_col_name = ['drug_id', 'drug_smile']
        self.p_col_name = ['prot_id','prot_seq']
        
        # the model pth 
        # ./savemodel/All-davis-Jan25_08-59-40.pth
        # ./savemodel/All-kiba-Jan25_09-05-47.pth
        # ./savemodel/All-metz-Jan25_09-05-01.pth
        self.model_fromTrain = './savemodel/All-kiba-Jan25_09-05-47.pth'        
                
        # model params
        self.drug_max_len = 100
        self.substructure_max_len = 100
        self.prot_max_len = 1022    # 1000, 1022
        self.mol2vec_dim = 300      # mol2vec:300, chemBERTa:384
        self.protvec_dim = 1280      # protvec:100,  esm2:1280
        
        self.latent_dim = 512     
        self.com_dim = 2048  
        self.mlp_dim = [1024, 512, 1]
        
        self.cuda = "0"
