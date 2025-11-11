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
        
        # "center_emb", "emb_length", "norm_emb": 
        # ['dataset', 'vec_dict', 'mat_dict', 'length_dict']
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'    #300 384 _drug_pretrain.pkl  _chemBERTa.pkl
        self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'  #100 1280 _prot_pretrain.pkl  _esm_pretrain.pkl           
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'   

        self.Learning_rate = 1e-4 # 1e-4
        self.Epoch = 200  # 200
        self.Batch_size = 256  # 16        
        self.max_patience = 20

        # model params
        self.drug_max_len = 100
        self.substructure_max_len = 100
        self.prot_max_len = 1022    # 1000, 1022
        self.mol2vec_dim = 300      # mol2vec:300, chemBERTa:384
        self.protvec_dim = 1280      # protvec:100,  esm2:1280
        
        self.latent_dim = 512     
        self.com_dim = 2048  # 预训练特征解压到2048
        self.mlp_dim = [1024, 512, 1]

        self.cuda = "1"  # Changed from "4" to "0" to match available GPU

    def set_dataset(self, data_name):
        self.dataset = data_name
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_chemBERTa.pkl' 
        self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'  
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'   
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'