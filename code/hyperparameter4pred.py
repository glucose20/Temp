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
        self.prot_max_len = 1022    # 1000, 1022 (ESM-C max: 2048)
        self.mol2vec_dim = 300      # mol2vec:300, chemBERTa:384
        
        # ESM-C (Cambrian) dimensions - Must match training config
        self.use_esmc = True  # Set to False to use ESM2
        self.esmc_model = "esmc_300m"  # Options: esmc_300m, esmc_600m, esmc_6b
        
        if self.use_esmc:
            # ESM-C dimensions
            if self.esmc_model == "esmc_300m":
                self.protvec_dim = 960   # ESM-C-300M: 960-dim
            elif self.esmc_model == "esmc_600m":
                self.protvec_dim = 1152  # ESM-C-600M: 1152-dim
            elif self.esmc_model == "esmc_6b":
                self.protvec_dim = 2560  # ESM-C-6B: 2560-dim
        else:
            # ESM2 (Legacy)
            self.protvec_dim = 1280  # ESM2: 1280-dim
        
        self.latent_dim = 512     
        self.com_dim = 2048  
        self.mlp_dim = [1024, 512, 1]
        
        self.cuda = "0"
