from datetime import datetime


class HyperParameter:
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        
        # Train on full dataset without fold split
        self.data_root = './data/dta-origin-dataset'  # Use original full dataset
        self.dataset = 'davis'      # davis, kiba, or metz
        self.running_set = 'all'    # 'all' means train on full dataset
        self.dataset_columns = ['drug_id', 'prot_id', 'label']
        self.is_esm = True
        
        # Pretrained feature paths
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'
        self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'
        
        # Training hyperparameters
        self.Learning_rate = 1e-4
        self.Epoch = 200  # Full training epochs
        self.Batch_size = 256
        self.max_patience = 20
        
        # Use 80% for train, 20% for validation (random split)
        self.train_ratio = 0.8
        self.valid_ratio = 0.2
        
        # Model parameters
        self.drug_max_len = 100
        self.substructure_max_len = 100
        self.prot_max_len = 1022    # 1000, 1022
        self.mol2vec_dim = 300      # mol2vec:300, chemBERTa:384
        self.protvec_dim = 1280     # protvec:100, esm2:1280
        
        self.latent_dim = 512
        self.com_dim = 2048
        self.mlp_dim = [1024, 512, 1]
        
        self.cuda = "0"  # GPU device

    def set_dataset(self, data_name):
        """Set dataset and update related paths"""
        self.dataset = data_name
        self.mol2vec_dir = f'./data/{self.dataset}/{self.dataset}_drug_pretrain.pkl'
        self.protvec_dir = f'./data/{self.dataset}/{self.dataset}_esm_pretrain.pkl'
        self.drugs_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_drugs.csv'
        self.prots_dir = f'{self.data_root}/{self.dataset}/{self.dataset}_prots.csv'
