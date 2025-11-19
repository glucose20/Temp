from datetime import datetime


class HyperParameter:
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.kfold = 5 

        self.data_root = './data/dta-5fold-dataset'  
        self.dataset = 'davis'      # davis kiba metz
        self.running_set = 'novel-pair'   # warm novel-drug novel-prot novel-pair
        self.dataset_columns = ['drug_id', 'prot_id', 'label']
        self.is_esm=True
        
        # ESM-C (Cambrian) configuration - Must be defined BEFORE using it
        self.use_esmc = True  # Set to False to use ESM2
        self.esmc_model = "esmc_300m"  # Options: esmc_300m, esmc_600m, esmc_6b
        
        # "center_emb", "emb_length", "norm_emb": 
        # ['dataset', 'vec_dict', 'mat_dict', 'length_dict']
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'    #300 384 _drug_pretrain.pkl  _chemBERTa.pkl
        
        # Protein embeddings path - ESM-C or ESM2
        if self.use_esmc:
            self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_pretrain.pkl'  # ESM-C embeddings
        else:
            self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'  # ESM2 embeddings (legacy)           
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'   

        self.Learning_rate = 1e-4 # 1e-4
        self.Epoch = 1  # 200
        self.Batch_size = 256  # 16        
        self.max_patience = 20

        # model params
        self.drug_max_len = 100
        self.substructure_max_len = 100
        self.prot_max_len = 1022    # 1000, 1022 (ESM-C max: 2048)
        self.mol2vec_dim = 300      # mol2vec:300, chemBERTa:384
        
        # ESM-C dimensions - Already configured above
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
        self.com_dim = 2048  # 预训练特征解压到2048
        self.mlp_dim = [1024, 512, 1]

        self.cuda = "1"  # Changed from "4" to "0" to match available GPU

    def set_dataset(self, data_name):
        self.dataset = data_name
        # self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_chemBERTa.pkl' 
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'
        
        # Update protein embeddings path based on ESM version
        if self.use_esmc:
            self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esmc_pretrain.pkl'  # ESM-C
        else:
            self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'  # ESM2
            
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'